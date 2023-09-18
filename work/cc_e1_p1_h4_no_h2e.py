import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def get_res_0(cc_obj, amp):
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e.vo)
    res +=     1.000000 * einsum('I,Iai->ai'         , cc_obj.h1p_eff, amp[1])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[0])
    res +=     1.000000 * einsum('Iai,I->ai'         , cc_obj.h1e1p.vo, amp[0])
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , cc_obj.h1e1p.oo, amp[1])
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , cc_obj.h1e1p.vv, amp[1])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , cc_obj.h1e1p.oo, amp[0], amp[0])
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , cc_obj.h1e1p.vv, amp[0], amp[0])
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , cc_obj.h1e1p.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , cc_obj.h1e1p.ov, amp[0], amp[1])
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , cc_obj.h1e1p.ov, amp[0], amp[1])
    res +=    -0.500000 * einsum('Ijb,bi,aj,I->ai'   , cc_obj.h1e1p.ov, amp[0], amp[0], amp[0])
    return res

def get_res_1(cc_obj, amp):
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('IJ,J->I'           , cc_obj.h1p, amp[0])
    res +=     1.000000 * einsum('ia,Iai->I'         , cc_obj.h1e.ov, amp[1])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1e1p.ov, amp[0])
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , cc_obj.h1e1p.ov, amp[0], amp[1])
    return res

def get_res_2(cc_obj, amp):
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1e1p.vo)
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , cc_obj.h1e.oo, amp[1])
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , cc_obj.h1e.vv, amp[1])
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , cc_obj.h1p, amp[1])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1e1p.oo, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1e1p.vv, amp[0])
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1e1p.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , cc_obj.h1e1p.oo, amp[0], amp[1])
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , cc_obj.h1e1p.vv, amp[0], amp[1])
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , cc_obj.h1e1p.ov, amp[1], amp[1])
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , cc_obj.h1e1p.ov, amp[1], amp[1])
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , cc_obj.h1e1p.ov, amp[1], amp[1])
    res +=    -0.500000 * einsum('Jjb,aj,J,Ibi->Iai' , cc_obj.h1e1p.ov, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('Jjb,bi,J,Iaj->Iai' , cc_obj.h1e1p.ov, amp[0], amp[0], amp[1])
    return res
