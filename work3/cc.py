import sys
import numpy, scipy

import pyscf
from pyscf import lib
from pyscf.lib import logger

class CoupledClusterAmplitudeSolverMixin(lib.StreamObject):
    ene_hf = None

    max_cycle = 200
    conv_tol = 1e-6
    verbose = 5

    is_converged = False

    def __init__(self):
        raise NotImplementedError

    def gen_func(self):
        raise NotImplementedError
    
    def get_init_amp(self):
        raise NotImplementedError

    def amp_to_vec(self, amp):
        raise NotImplementedError

    def vec_to_amp(self, vec):
        raise NotImplementedError

    def res_to_vec(self, res):
        raise NotImplementedError

    def vec_to_res(self, vec):
        raise NotImplementedError
    
    def kernel(self, amp=None):
        raise NotImplementedError

# Can we inherent from different solver? Like DIIS and NewtonKrylov.
class NewtonKrylov(CoupledClusterAmplitudeSolverMixin):
    def kernel(self, amp=None):
        # The function returns a tuple containing the following values:
        #   - ene_sol: The final total energy.
        #   - ene_sol - ene_hf: The final correlation energy.
        #   - amp_sol: The final amplitudes.
        #
        # The kernel function uses the Newton-Krylov method from the scipy.optimize module
        # to solve the amplitude equations. The method requires a residual function,
        # which is generated using the gen_res_func method. The function also calculates
        # the mean-field energy, initial correlation energy, and initial total energy
        # using the get_ene_hf and get_ene_cor methods.
        
        # Record the initial CPU and wall time
        cput0 = (logger.process_clock(), logger.perf_counter())
        # Create a new logger instance
        log   = logger.new_logger(self)
        
        # If no initial amplitudes are provided, generate them
        if amp is None:
            amp = self.get_init_amp()
            log.timer('Time to generate initial amplitudes', *cput0)

        # Calculate the mean-field energy
        assert self.ene_hf is not None
        ene_hf   = self.ene_hf

        func = self.gen_func()

        # Convert the amplitudes to a vector
        vec_init = self.amp_to_vec(amp)
        # Calculate the initial residuals
        ene_init, res_init = func(vec_init, verbose=False)

        # Log the initial energy values and residual norm
        log.info('\nMean-field energy          = % 12.8f', ene_hf)
        log.info('Initial correlation energy = % 12.8f', ene_init - ene_hf)
        log.info('Initial total energy       = % 12.8f', ene_init)
        log.info('Initial residual norm      = % 12.4e\n', numpy.linalg.norm(res_init))

        # Import the optimize module from scipy
        from scipy import optimize
        try:
            # Solve the amplitude equations using the Newton-Krylov method
            vec_sol = optimize.newton_krylov(
                lambda x: func(x)[1], 
                vec_init,  verbose=0, 
                f_tol=self.conv_tol, 
                maxiter=self.max_cycle,
                )
                
        except optimize.nonlin.NoConvergence as e:
            # If the method does not converge, log a warning and use the last solution
            log.warn('Newton-Krylov method did not converge')
            vec_sol = e.args[0]
        
        # Convert the solution vector back to amplitudes
        amp_sol = self.vec_to_amp(vec_sol)

        # Log the time taken to solve the amplitude equations
        log.timer('\nTime to solve coupled-cluster amplitude equations', *cput0)

        # Dump final results
        ene_sol, res = func(vec_sol, verbose=False)

        # Log the final energy values and residual norm
        log.info('\nFinal correlation energy    = % 12.8f', ene_sol + ene_hf)
        log.info('Final total energy          = % 12.8f',   ene_sol)
        log.info('Final residual norm         = % 12.4e',   numpy.linalg.norm(res))

        self.is_converged = (numpy.linalg.norm(res) < self.conv_tol)

        # Return the final total energy, correlation energy, and amplitudes
        return ene_sol, ene_sol - ene_hf, amp_sol
    
class FromPySCF(CoupledClusterAmplitudeSolverMixin):
    def __init__(self, mf):
        self.mf = mf # For RCCSD and GCCSD
        self._base = pyscf.cc.CCSD(mf)

        self.ene_hf = self._base.get_e_hf()
        self._eris = self._base.ao2mo()
        self._eris.mo_energy = self._eris.fock.diagonal().copy()
        self._eris.fock += numpy.diag(self._eris.mo_energy)

    def get_init_amp(self):
        return self._base.get_init_guess()
    
    def gen_func(self):
        log = logger.new_logger(self)

        global iter_cc
        iter_cc = 0

        # Note that the ERIs is modified
        _eris = self._eris

        # Check the diagonal of the Fock matrix
        mo_ene = _eris.mo_energy
        assert numpy.linalg.norm(mo_ene - 0.5 * _eris.fock.diagonal()) < 1e-12

        nocc = self._base.nocc
        nvir = self._base.nmo - nocc
        mo_e_o = mo_ene[:nocc]
        mo_e_v = mo_ene[nocc:]

        eia = mo_e_o[:,None] - mo_e_v
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)

        def func(vec, verbose=True):
            t1e, t2e = self.vec_to_amp(vec)

            cc_obj = self._base
            ene = cc_obj.energy(t1=t1e, t2=t2e, eris=_eris)
            rr  = cc_obj.update_amps(t1e, t2e, eris=self._eris)
            r1e = rr[0] * eia
            r2e = rr[1] * eijab

            res = self.res_to_vec((r1e, r2e))

            if verbose:
                global iter_cc
                iter_cc += 1

                log.info(
                    'CCSD iter %4d, energy = %12.8f, residual = %12.4e',
                    iter_cc, ene, numpy.linalg.norm(res)
                    )

            return ene, res

        return func
    
    def amp_to_vec(self, amp):
        t1e, t2e = amp
        return self._base.amplitudes_to_vector(t1e, t2e)
    
    def vec_to_amp(self, vec):
        return self._base.vector_to_amplitudes(vec)
    
    def res_to_vec(self, res):
        r1e, r2e = res
        return self._base.amplitudes_to_vector(r1e, r2e)
    
    def vec_to_res(self, vec):
        return self._base.vector_to_amplitudes(vec)

if __name__ == '__main__':
    import numpy
    from pyscf import gto, scf, cc

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., 1., 0.)],
        ['H', (0., 0., 1.)],
    ]
    mol.basis = '631g*'
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()
    
    class Solver(FromPySCF, NewtonKrylov):
        pass

    cc_obj = Solver(mf)
    cc_obj.conv_tol = 1e-8
    ene_sol = cc_obj.kernel()[0]
    ene_ref = cc_obj._base.run(verbose=4, conv_tol=1e-8).e_corr

    assert abs(ene_sol - ene_ref) < 1e-8

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()
        
    cc_obj = Solver(mf)
    cc_obj.conv_tol = 1e-8
    ene_sol = cc_obj.kernel()[0]
    ene_ref = cc_obj._base.run(verbose=4, conv_tol=1e-8).e_corr

    assert abs(ene_sol - ene_ref) < 1e-8