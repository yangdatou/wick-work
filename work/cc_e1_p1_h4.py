import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def get_ene(cc_obj, amp):
    res  =     1.000000 * einsum('ia,ai->'           , cc_obj.h1e.ov, amp[0])
    res +=     1.000000 * einsum('ia,ai->'           , cc_obj.h1e.ov, amp[1])
    return res

def get_res_bra_0(cc_obj, amp):
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e.vo)
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[0])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[1])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[1])
    res +=    -1.000000 * einsum('jb,aj,bi->ai'      , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[1], amp[1])
    return res

def get_res_bra_1(cc_obj, amp):
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1e1p.ov, amp[0])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1e1p.ov, amp[1])
    return res

def get_res_bra_2(cc_obj, amp):
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1e1p.vo)
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1e1p.oo, amp[0])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1e1p.oo, amp[1])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1e1p.vv, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1e1p.vv, amp[1])
    res +=    -1.000000 * einsum('Ijb,aj,bi->Iai'    , cc_obj.h1e1p.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1e1p.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1e1p.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1e1p.ov, amp[1], amp[1])
    return res
