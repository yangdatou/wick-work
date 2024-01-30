from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, braE2, commute

H = one_e("D", ["occ", "vir"], norder=True)

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)

bra = braE1("occ", "vir")
S = bra*(H + HT + Fraction('1/2')*HTT ) # Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)

bra = braE2("occ", "vir", "occ", "vir")
S = bra*(H + HT + Fraction('1/2')*HTT ) # Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
