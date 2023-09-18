# Standard Library Imports
import os, sys

# Extend system path to include custom libraries
sys.path.extend([
    "../wick-main/",
    "../cceph-main/",
    "../cqcpy-master/",
    "../epcc-hol/"
])

# Wick module imports
from fractions import Fraction
from math import factorial
from itertools import permutations
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import (one_e, two_e, one_p, two_p, ep11, P1, EPS1, E1, E2,
                              braE1, braE2, ketE1, ketE2, braP1, braP1E1, commute)

from wick.convenience import (Idx, Sigma, Tensor, FOperator, Term, get_sym, Expression)
from wick.operator import Projector, BOperator, FOperator
from wick.operator import TensorSym, Tensor, Sigma, normal_ordered

LOG_TMPDIR = os.environ.get("LOG_TMPDIR", "/scratch/global/yangjunjie/")

# Constants
PYTHON_FILE_TAB = "    "  # Define tab spacing for Python file output

def space_idx_formatter(name, space_list):
    """Format space index based on given rules."""

    spaces = [space[0] for space in space_list if space != "nm"]
    return f"{name}." + "".join(spaces) if spaces else f"{name}"


def einsum_str(t_obj):
    """Generate string for numpy einsum function based on tensor object."""

    imap = t_obj._idx_map()
    scalar_str = f"{float(t_obj.scalar): 12.6f}"
    final_str, index_str, tensor_str = "", "", ""

    for tensor in t_obj.tensors:
        if not tensor.name:  # Part of final string index
            final_str += tensor._istr(imap)
        else:
            name = ", " + (space_idx_formatter(tensor.name, [idx.space for idx in
                                                             tensor.indices]) if "amp" not in tensor.name else tensor.name)
            tensor_str += name
            index_str += tensor._istr(imap) + ","

    einsum_input = f"'{index_str[:-1]}->{final_str}'"
    return f"{scalar_str} * einsum({einsum_input:20s}{tensor_str})"


def print_einsum(expr_obj):
    """Print einsum representation for given expression object."""

    equations = []
    for idx, term in enumerate(expr_obj.terms):
        lhs = "res  = " if idx == 0 else "res += "
        equation = f"{lhs}{einsum_str(term)}"
        equations.append(equation)

    return "\n".join(equations)


def gen_einsum_fxn(expr, name_str):
    """Generate Python function for numpy einsum based on expression and write to file."""

    # Construct the function string
    function_lines = [f"def {name_str}(cc_obj, amp):"]
    function_lines.extend([PYTHON_FILE_TAB + line for line in print_einsum(expr).split("\n")])
    function_lines.append(PYTHON_FILE_TAB + "return res")

    return '\n'.join(function_lines)

def PN(ph_max, name):
    """
    Return the tensor representation of a Boson ph_max-excitation operator for spaces=["mn"]

    Args:
    - ph_max (int): Maximum number of excitations.
    - name (string): Name of the tensor.
    - index_key (optional): Additional key for index, defaults to None.

    Returns:
    - Expression: Representation of the ph_max-excitation operator.
    """

    # List to store the terms of the ph_max-excitation operator
    terms = []

    # Generate the symmetries for the tensor, which are all the permutations of indices from 0 to ph_max-1
    all_permutations = list(permutations(range(ph_max)))
    sym = TensorSym(all_permutations, [1] * factorial(ph_max))

    # Construct the list of indices for "mn" space.
    indices = [Idx(i, "nm", fermion=False) for i in range(ph_max)]

    # Create the summation indices (Sigma) for each index
    sums = [Sigma(idx) for idx in indices]

    # Define the tensors using the indices and the name provided, with the defined symmetry
    tensors = [Tensor(indices, name, sym=sym)]

    # Define the Boson operators for each index
    operators = [BOperator(idx, True) for idx in indices]

    # Compute the prefactor as 1 over factorial of ph_max
    s = Fraction(1, factorial(ph_max))

    # Construct the term for the ph_max-excitation operator using the prefactor, summation indices, tensors, and operators
    term = Term(s, sums, tensors, operators, [], index_key=None)

    # Add the term to the list of terms
    terms.append(term)

    # Return the entire expression for the ph_max-excitation operator
    return Expression(terms)


def PNE1(ph_max, name):
    """
    Return the tensor representation of a coupled Fermion-ph_max Boson excitation operator
    for bspaces=["mn"], ospaces=["ij"], and vspaces=["ab"].

    Args:
    - ph_max (int): Maximum number of Boson excitations.
    - name (string): Name of the tensor.
    - index_key (optional): Additional key for index, defaults to None.

    Returns:
    - Expression: Representation of the coupled Fermion-ph_max Boson excitation operator.
    """

    # List to store the terms of the operator
    terms = []

    # Generate the symmetries for the tensor, which are permutations of Boson indices
    boson_perms = list(permutations(range(ph_max)))
    full_perms = [x + (ph_max, ph_max + 1) for x in boson_perms]
    sym = TensorSym(full_perms, [1] * factorial(ph_max))

    # Create the Boson indices
    b_indices = [Idx(i, "nm", fermion=False) for i in range(ph_max)]

    # Create the Fermion indices
    i = Idx(0, "occ")
    a = Idx(0, "vir")

    # Combine all indices
    all_indices = b_indices + [a, i]

    # Create the summation indices (Sigma) for each index
    sums = [Sigma(idx) for idx in all_indices]

    # Define the tensors using the indices and the name provided, with the defined symmetry
    tensors = [Tensor(all_indices, name, sym=sym)]

    # Define the Boson and Fermion operators for each index
    operators = [BOperator(idx, True) for idx in b_indices] + [FOperator(a, True), FOperator(i, False)]

    # Compute the prefactor as 1 over factorial of ph_max
    s = Fraction(1, factorial(ph_max))

    # Construct the term for the operator using the prefactor, summation indices, tensors, and operators
    term = Term(s, sums, tensors, operators, [], index_key=None)

    # Add the term to the list of terms
    terms.append(term)

    # Return the entire expression for the operator
    return Expression(terms)

def braPN(ph_max):
    """
    Return projection onto space of ph_max Boson excitations

    Args:
    - ph_max (int): Number of Boson excitations.
    - space (str): Name of boson space.
    - index_key (optional): Additional key for index, defaults to None.

    Returns:
    - Expression: Projection onto space of ph_max Boson excitations.
    """

    # Create ph_max Boson indices for the given space
    b_indices = [Idx(i, "nm", fermion=False) for i in range(ph_max)]

    # Create Boson operators for each index
    operators = [BOperator(idx, False) for idx in b_indices]

    # Define the tensors using the indices
    tensors = [Tensor(b_indices, "")]

    # Construct the term for the operator using the tensors and operators
    term = Term(1, [], tensors, operators, [], index_key=None)

    # Return the entire expression for the operator
    return Expression([term])

def braPNE1(ph_max):
    """
    Return left-projector onto a space of single excitations coupled to
    ph_max boson excitations.

    ph_max (int): Number of boson excitations.
    bspace (str): boson space
    ospace (str): occupied space
    vspace (str): virtual space
    """

    # Create ph_max Boson indices for the given space
    b_indices = [Idx(i, "nm", fermion=False) for i in range(ph_max)]

    # Create Fermion indices
    i = Idx(0, "occ")
    a = Idx(0, "vir")

    # Combine all indices
    all_indices = b_indices + [a, i]

    # Create Boson and Fermion operators for each index
    operators = [BOperator(idx, False) for idx in b_indices] + [FOperator(i, True), FOperator(a, False)]

    # Define the tensors using the indices
    tensors = [Tensor(all_indices, "")]

    # Construct the term for the operator using the tensors and operators
    term = Term(1, [], tensors, operators, [], index_key=None)

    # Return the entire expression for the operator
    return Expression([term])

def gen_epcc_eqs(with_h2e=False, elec_order=2, ph_order=1, hbar_order=4):
    name = "cc_e%d_p%d_h%d" % (elec_order, ph_order, hbar_order) + ("_with_h2e" if with_h2e else "_no_h2e")
    log = sys.stdout

    H1e   = one_e("cc_obj.h1e", ["occ", "vir"], norder=True)
    H2e   = two_e("cc_obj.h2e", ["occ", "vir"], norder=True, compress=True)
    H1p   = one_p("cc_obj.h1p_eff") + two_p("cc_obj.h1p")
    H1e1p = ep11("cc_obj.h1e1p", ["occ", "vir"], ["nm"], norder=True)
    H = H1e + H1p + H1e1p if not with_h2e else H1e + H2e + H1p + H1e1p

    log.write("Finishing Building Hamiltonian....\n")

    if elec_order == 1:
        T = E1("amp[0]", ["occ"], ["vir"])
        bra_list = [braE1("occ", "vir")]

    elif elec_order == 2:
        T = E1("amp[0]", ["occ"], ["vir"]) + E2("amp[1]", ["occ"], ["vir"])
        bra_list = [braE1("occ", "vir"), braE2("occ", "vir", "occ", "vir")]

    else:
        raise Exception("elec_order must be 1 or 2")

    for i in range(ph_order):
        T += PN(i+1, "amp[%d]" % (elec_order + 2 * i - 1))
        T += PNE1(i+1, "amp[%d]" % (elec_order + 2 * i))

    log.write("Finishing Building T....\n")

    Hbar = [H]
    for ihbar in range(1, hbar_order + 1):
        hbar = commute(Hbar[-1], T) * Fraction(1, factorial(ihbar))
        Hbar.append(hbar)

    log.write("Finishing Building Hbar....\n")

    for i in range(1, ph_order + 1):
        bra_list.append(braPN(i))
        bra_list.append(braPNE1(i))

    log.write("Finishing Initialization....\n")

    def gen_res_func(ih, ibra):
        log.write("ibra = %d, ih = %d\n" % (ibra, ih))
        h   = Hbar[ih]
        bra = bra_list[ibra]

        out = apply_wick(bra * h)
        out.resolve()

        tmp = AExpression(Ex=out)

        log.write("tmp = \n")
        log.write(str(tmp))
        return tmp

    res = "import numpy, functools\neinsum = functools.partial(numpy.einsum, optimize=True)\n"

    for ibra, bra in enumerate(bra_list):
        final = None

        for ih, h in enumerate(Hbar):
            tmp = gen_res_func(ih, ibra)
            final = tmp if final is None else final + tmp

            if len(tmp.terms) <= 1 and ih > 0:
                res += "\n" + gen_einsum_fxn(final, f"get_res_{ibra}") + "\n"
                break

            if ih == hbar_order:
                raise Exception("bra %d did not converge" % ibra)

    log.write("Finishing Building Residuals....\n")
    log.write("Final Expression = \n")
    log.write(res)

    with open(name + ".py", "w") as f:
        f.write(res)

if __name__ == "__main__":
    gen_epcc_eqs(elec_order=2, ph_order=1, hbar_order=5, with_h2e=True)
    gen_epcc_eqs(elec_order=2, ph_order=2, hbar_order=5, with_h2e=True)

