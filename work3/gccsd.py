import pyscf
from pyscf import gto, scf, lib
from pyscf import cc as ccsd
from pyscf.lib import logger

from cc import CoupledClusterAmplitudeSolverMixin

class GeneralSpinCoupledClusterSingleDoubleFromWick(CoupledClusterAmplitudeSolverMixin):
    def __init__(self, mf):
        self.mf = mf

        is_ghf  = isinstance(mf, scf.ghf.GHF)
        assert is_ghf and mf.converged

        self._base = pyscf.cc.CCSD(mf)
        self.ene_hf = self._base.get_e_hf()

    def get_init_amp(self):
        t1e, t2e = self._base.get_init_guess()
        t1e_vo = t1e.transpose(1, 0)
        t2e_vvoo = t2e.transpose(2, 3, 0, 1)
        return t1e_vo, t2e_vvoo
    
    def amp_to_vec(self, amp):
        t1e_vo, t2e_vvoo = amp
        t1e = t1e_vo.transpose(1, 0)
        t2e = t2e_vvoo.transpose(2, 3, 0, 1)
        return self._base.amplitudes_to_vector(t1e, t2e)
    
    def vec_to_amp(self, vec):
        t1e, t2e = self._base.vector_to_amplitudes(vec)
        t1e_vo   = t1e.transpose(1, 0)
        t2e_vvoo = t2e.transpose(2, 3, 0, 1)
        return t1e_vo, t2e_vvoo
    
    def res_to_vec(self, res):
        norb = self._base.nmo
        nocc = self._base.nocc
        nvir = norb - nocc

        r1e_vo, r2e_vvoo = res
        assert r1e_vo.shape == (nvir, nocc)
        assert r2e_vvoo.shape == (nvir, nvir, nocc, nocc)

        r1e_ov = r1e_vo.transpose(1, 0)
        r2e_oovv = r2e_vvoo.transpose(2, 3, 0, 1)
        return self._base.amplitudes_to_vector(r1e_ov, r2e_oovv)
    
    def vec_to_res(self, vec):
        norb = self._base.nmo
        nocc = self._base.nocc
        nvir = norb - nocc

        r1e_ov, r2e_oovv = self._base.vector_to_amplitudes(vec)
        assert r1e_ov.shape == (nocc, nvir)
        assert r2e_oovv.shape == (nocc, nocc, nvir, nvir)

        r1e_vo = r1e_ov.transpose(1, 0)
        r2e_vvoo = r2e_oovv.transpose(2, 3, 0, 1)
        return r1e_vo, r2e_vvoo 
    
class WithFullERIS(GeneralSpinCoupledClusterSingleDoubleFromWick):           
    def gen_func(self):
        log = logger.new_logger(self)

        _eris = self._base.ao2mo()
        coeff = _eris.mo_coeff

        nao, norb = coeff.shape
        nocc = self._base.nocc
        nvir = norb - nocc

        # Note that the ERIs is modified
        class CoupledClusterProblem(object):
            pass

        class H1eBlocks(object):
            pass

        class H2eBlocks(object):
            pass

        h1e = H1eBlocks()
        h1e.oo = _eris.fock[:nocc, :nocc].copy()
        h1e.vv = _eris.fock[nocc:, nocc:].copy()
        h1e.ov = _eris.fock[:nocc, nocc:].copy()
        h1e.vo = _eris.fock[nocc:, :nocc].copy()

        from pyscf import ao2mo
        assert self._base._scf._eri is not None
        coeff_alph = coeff[:nao//2]
        coeff_beta = coeff[nao//2:]
        eris  = ao2mo.kernel(self._base._scf._eri, coeff_alph)
        eris += ao2mo.kernel(self._base._scf._eri, coeff_beta)
        eri1  = ao2mo.kernel(
            self._base._scf._eri, 
            (coeff_alph, coeff_alph, coeff_beta, coeff_beta)
            )
        eris += eri1
        eris += eri1.T

        # TODO: transform into Chemist's index
        eris = ao2mo.restore(1, eris, norb).reshape(norb, norb, norb, norb)
        eris = eris.transpose(0, 2, 1, 3)

        h2e = H2eBlocks()
        h2e.oooo = eris[:nocc, :nocc, :nocc, :nocc].copy()
        h2e.ovoo = eris[:nocc, nocc:, :nocc, :nocc].copy()
        h2e.ovvo = eris[:nocc, nocc:, nocc:, :nocc].copy()
        h2e.ovov = eris[:nocc, nocc:, :nocc, nocc:].copy()
        h2e.oovv = eris[:nocc, :nocc, nocc:, nocc:].copy()
        h2e.ovvv = eris[:nocc, nocc:, nocc:, nocc:].copy()
        h2e.vvvv = eris[nocc:, nocc:, nocc:, nocc:].copy()
        h2e.ooov = eris[:nocc, :nocc, :nocc, nocc:].copy()
        h2e.ooov = eris[:nocc, :nocc, :nocc, nocc:].copy()
        h2e.vvoo = eris[nocc:, nocc:, :nocc, :nocc].copy()
        h2e.vvov = eris[nocc:, nocc:, :nocc, nocc:].copy()

        cc_obj = CoupledClusterProblem()
        cc_obj.h1e = h1e
        cc_obj.h2e = h2e

        global iter_cc
        iter_cc = 0

        from cc_e2_h4_with_h2e import res_ibra_0 as ccsd_r1e
        from cc_e2_h4_with_h2e import res_ibra_1 as ccsd_r2e

        def func(vec, verbose=True):
            amp = self.vec_to_amp(vec)

            ene = self._base.energy(
                t1=amp[0].transpose(1, 0),
                t2=amp[1].transpose(2, 3, 0, 1),
                eris=_eris
                )
            r1e = ccsd_r1e(cc_obj=cc_obj, amp=amp)
            r2e = ccsd_r2e(cc_obj=cc_obj, amp=amp)
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

        
if __name__ == "__main__":
    import numpy
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., 1., 0.)],
        ['H', (0., 0., 1.)],
    ]
    mol.basis = 'sto3g'
    mol.build()

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    from cc import NewtonKrylov, FromPySCF
    class Solver(FromPySCF, NewtonKrylov):
        pass

    s1 = Solver(mf)
    amp0 = s1.get_init_amp()
    vec0 = s1.amp_to_vec(amp0)
    f1   = s1.gen_func()
    e1, r1 = f1(vec0)
    r1_1e, r1_2e = s1.vec_to_res(r1)
    r1_1e = r1_1e.transpose(1, 0)
    r1_2e = r1_2e.transpose(2, 3, 0, 1)

    from cc import NewtonKrylov
    class Solver(WithFullERIS, NewtonKrylov):
        pass

    s2 = Solver(mf)
    ecor_sol = s2.kernel()[0]

    ecor_ref = s2._base.run(verbose=5).e_corr
    assert abs(ecor_sol - ecor_ref) < 1e-8