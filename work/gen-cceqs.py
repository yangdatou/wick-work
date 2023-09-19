import sys, os
sys.path.append("../wick-main/")

from fractions import Fraction
from math import factorial
from itertools import permutations

import wick

from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import (
    one_e, two_e, one_p, two_p, ep11, 
    E1, E2, braE1, braE2, commute,
    Idx, Sigma, Tensor, FOperator, 
    Term, Expression
)

BraE1 = braE1
BraE2 = braE2

from wick.operator import BOperator, FOperator
from wick.operator import TensorSym, Tensor, Sigma

# Constants
PYTHON_FILE_TAB = "    "  # Define tab spacing for Python file output

def get_info():
    import platform
    import subprocess, socket
    from datetime import datetime

    # Get machine information
    machine_info = f"{platform.system()} {platform.release()}"
    hostname = f"{socket.gethostname()}"

    # Get Git information (if available)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        git_commit = None
        git_branch = None

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    comment  = f"Generated by {os.path.basename(__file__)} at {timestamp}\n" 
    comment += PYTHON_FILE_TAB + f"Machine Info: {machine_info}\n"
    comment += PYTHON_FILE_TAB + f"Hostname:     {hostname}\n"
    if git_branch:
        comment += PYTHON_FILE_TAB + f"Git Branch:   {git_branch}\n"

    if git_commit:
        comment += PYTHON_FILE_TAB + f"Git Commit:   {git_commit}\n"

    return comment

def term_info(expr, name=None):
    if name is None:
        name = ""

    info = ""

    if expr is None:
        return ""
    
    for term in expr.terms:
        assert len(term.tensors) == 1
        tensor = term.tensors[0]
        operators = []

        tname = tensor.name
        if len(tname) == 0:
            tname = name
        
        if len(tname) > 0:
            tname = PYTHON_FILE_TAB + tname + ":"
        
        for op in term.operators:
            space = "" if op.idx.space == "nm" else "_" + op.idx.space[0]
            op_old = str(op)
            op_new = op_old[0] + ("+" if "^{\\dagger}" in op_old else "")
            operators.append(
                op_new + space
            )
        
        info += "%s %s" % (tname, " ".join(operators)) + "\n"
    return info

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
            name = ", " + (space_idx_formatter(
                tensor.name, [idx.space for idx in tensor.indices]) \
                    if "amp" not in tensor.name else tensor.name
                )
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


def gen_einsum_fxn(expr, name_str, comment=None):
    """Generate Python function for numpy einsum based on expression and write to file.

    Args:
        expr (AExpression): The expression representing the einstein summation.
        name_str (str): The name of the generated function.
        comment (str, optional): Additional comments to include in the function. Default is None.

    Returns:
        str: The generated Python function as a string.

    """

    # Construct the function string
    function_lines = [f"def {name_str}(cc_obj, amp):"]
    
    if comment:
        function_lines.append(PYTHON_FILE_TAB + '"""')
        function_lines.append(PYTHON_FILE_TAB + comment)
        function_lines.append(PYTHON_FILE_TAB + '"""')

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

def BraPN(ph_max):
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

def BraPNE1(ph_max):
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
    H1p1e = ep11("cc_obj.h1p1e", ["occ", "vir"], ["nm"], norder=True)
    H = H1e + H1p + H1p1e if not with_h2e else H1e + H2e + H1p + H1p1e

    log.write("Finishing Building Hamiltonian....\n")
    log.flush()

    bra_list = []
    if elec_order == 1:
        T = E1("amp[0]", ["occ"], ["vir"])
        bra_list += [braE1("occ", "vir")]

    elif elec_order == 2:
        T = E1("amp[0]", ["occ"], ["vir"]) + E2("amp[1]", ["occ"], ["vir"])
        bra_list += [braE1("occ", "vir"), braE2("occ", "vir", "occ", "vir")]

    else:
        raise Exception("elec_order must be 1 or 2")

    for i in range(ph_order):
        T += PN(i+1, "amp[%d]" % (elec_order + 2 * i))
        T += PNE1(i+1, "amp[%d]" % (elec_order + 2 * i + 1))

    amp_info = term_info(T)
    log.write("Finishing Building T....\n")
    log.flush()

    Hbar = [H]
    for ihbar in range(1, hbar_order + 1):
        hbar = commute(Hbar[-1], T) * Fraction(1, factorial(ihbar))
        # hbar.resolve()
        Hbar.append(hbar)

    log.write("Finishing Building Hbar....\n")
    log.flush()

    for i in range(1, ph_order + 1):
        bra_list.append(BraPN(i))
        bra_list.append(BraPNE1(i))
    bra_list += [None]

    log.write("Finishing Initialization....\n")
    log.write("Number of terms in amplitude   = % 2d\n" % (len(T.terms)))
    log.write("Number of terms of bra_list    = % 2d\n" % (len(bra_list)))
    log.write("Number of terms of Hbar        = % 2d\n" % (len(Hbar)))
    
    res = "import numpy, functools\neinsum = functools.partial(numpy.einsum, optimize=True)\n"

    func_list = []

    # Iterate over bra_list and Hbar
    for ibra, bra in enumerate(bra_list):
        if bra is not None:
            func_name = f"res_{ibra}"
        else:
            func_name = "ene"

        log.write("\n\nGenerating %s.%s ..." % (name, func_name))

        is_converged = False
        comment = get_info() + "\n" + amp_info 
        if "res" in func_name:
            comment += PYTHON_FILE_TAB + "res   : %s" % term_info(bra, name=None)

        final = AExpression()

        for ih, h in enumerate(Hbar):
            if bra is not None:
                out = apply_wick(bra * h)
            else:
                out = apply_wick(h)
            out.resolve()

            tmp = AExpression(Ex=out)
            final += tmp

            # Logging details
            comment_line = "ih = %d, ibra = %d, len(tmp.terms) = %d" % (ih, ibra, len(tmp.terms))
            comment += "\n" + PYTHON_FILE_TAB + comment_line
            log.write("\n" + comment_line)

            if len(tmp.terms) == 0 and ih > 0:
                is_converged = True
                break

        if not is_converged:
            log.write("\nibra = %d, is not converged up to hbar_order = %d\n" % (ibra, hbar_order))
            comment += "\n\n" + PYTHON_FILE_TAB + "NOTE: the equation is not truncted."

        func_list.append(
            gen_einsum_fxn(
                final, func_name, comment=comment
            )
        )

    res += "\n" + '\n\n'.join(func_list) + "\n"
    log.write("\n\n" + res)

    with open(name + ".py", "w") as f:
        f.write(res)

if __name__ == "__main__":
    elec_order = int(sys.argv[1])
    ph_order   = int(sys.argv[2])
    hbar_order = int(sys.argv[3])
    with_h2e   = bool(sys.argv[4])

    gen_epcc_eqs(elec_order=elec_order, ph_order=ph_order, hbar_order=hbar_order, with_h2e=with_h2e)

