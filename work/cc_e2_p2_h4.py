import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def get_ene(cc_obj, amp):
    res  =     1.000000 * einsum('I,I->'             , cc_obj.h1p_eff, amp[1])
    res +=     1.000000 * einsum('ia,ai->'           , cc_obj.h1e.ov, amp[0])
    res +=     1.000000 * einsum('Iia,Iai->'         , cc_obj.h1e1p.ov, amp[2])
    res +=     1.000000 * einsum('Iia,ai,I->'        , cc_obj.h1e1p.ov, amp[0], amp[1])
    return res

def get_res_bra_0(cc_obj, amp):
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e.vo)
    res +=     1.000000 * einsum('I,Iai->ai'         , cc_obj.h1p_eff, amp[2])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[0])
    res +=     1.000000 * einsum('Iai,I->ai'         , cc_obj.h1e1p.vo, amp[1])
    res +=    -1.000000 * einsum('jb,abji->ai'       , cc_obj.h1e.ov, amp[1])
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , cc_obj.h1e1p.oo, amp[2])
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , cc_obj.h1e1p.vv, amp[2])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , cc_obj.h1e1p.oo, amp[0], amp[1])
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , cc_obj.h1e1p.vv, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,abji,I->ai'    , cc_obj.h1e1p.ov, amp[1], amp[1])
    res +=    -0.500000 * einsum('Ijb,bi,aj,I->ai'   , cc_obj.h1e1p.ov, amp[0], amp[0], amp[1])
    return res

def get_res_bra_1(cc_obj, amp):
    res  =     1.000000 * einsum('ki,bakj->abij'     , cc_obj.h1e.oo, amp[1])
    res +=    -1.000000 * einsum('kj,baki->abij'     , cc_obj.h1e.oo, amp[1])
    res +=     1.000000 * einsum('ac,bcji->abij'     , cc_obj.h1e.vv, amp[1])
    res +=    -1.000000 * einsum('bc,acji->abij'     , cc_obj.h1e.vv, amp[1])
    res +=     1.000000 * einsum('Iai,Ibj->abij'     , cc_obj.h1e1p.vo, amp[2])
    res +=    -1.000000 * einsum('Iaj,Ibi->abij'     , cc_obj.h1e1p.vo, amp[2])
    res +=    -1.000000 * einsum('Ibi,Iaj->abij'     , cc_obj.h1e1p.vo, amp[2])
    res +=     1.000000 * einsum('Ibj,Iai->abij'     , cc_obj.h1e1p.vo, amp[2])
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Iki,ak,Ibj->abij'  , cc_obj.h1e1p.oo, amp[0], amp[2])
    res +=     1.000000 * einsum('Iki,bk,Iaj->abij'  , cc_obj.h1e1p.oo, amp[0], amp[2])
    res +=     1.000000 * einsum('Iki,bakj,I->abij'  , cc_obj.h1e1p.oo, amp[1], amp[1])
    res +=     1.000000 * einsum('Ikj,ak,Ibi->abij'  , cc_obj.h1e1p.oo, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ikj,bk,Iai->abij'  , cc_obj.h1e1p.oo, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ikj,baki,I->abij'  , cc_obj.h1e1p.oo, amp[1], amp[1])
    res +=     1.000000 * einsum('Iac,ci,Ibj->abij'  , cc_obj.h1e1p.vv, amp[0], amp[2])
    res +=    -1.000000 * einsum('Iac,cj,Ibi->abij'  , cc_obj.h1e1p.vv, amp[0], amp[2])
    res +=     1.000000 * einsum('Iac,bcji,I->abij'  , cc_obj.h1e1p.vv, amp[1], amp[1])
    res +=    -1.000000 * einsum('Ibc,ci,Iaj->abij'  , cc_obj.h1e1p.vv, amp[0], amp[2])
    res +=     1.000000 * einsum('Ibc,cj,Iai->abij'  , cc_obj.h1e1p.vv, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ibc,acji,I->abij'  , cc_obj.h1e1p.vv, amp[1], amp[1])
    res +=     1.000000 * einsum('Ikc,acji,Ibk->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ikc,acki,Ibj->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=     1.000000 * einsum('Ikc,ackj,Ibi->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ikc,baki,Icj->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=     1.000000 * einsum('Ikc,bakj,Ici->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ikc,bcji,Iak->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=     1.000000 * einsum('Ikc,bcki,Iaj->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ikc,bckj,Iai->abij', cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=    -0.500000 * einsum('Ikc,ak,bcji,I->abij', cc_obj.h1e1p.ov, amp[0], amp[1], amp[1])
    res +=     0.500000 * einsum('Ikc,bk,acji,I->abij', cc_obj.h1e1p.ov, amp[0], amp[1], amp[1])
    res +=    -0.500000 * einsum('Ikc,ci,ak,Ibj->abij', cc_obj.h1e1p.ov, amp[0], amp[0], amp[2])
    res +=     0.500000 * einsum('Ikc,ci,bk,Iaj->abij', cc_obj.h1e1p.ov, amp[0], amp[0], amp[2])
    res +=     0.500000 * einsum('Ikc,ci,bakj,I->abij', cc_obj.h1e1p.ov, amp[0], amp[1], amp[1])
    res +=     0.500000 * einsum('Ikc,cj,ak,Ibi->abij', cc_obj.h1e1p.ov, amp[0], amp[0], amp[2])
    res +=    -0.500000 * einsum('Ikc,cj,bk,Iai->abij', cc_obj.h1e1p.ov, amp[0], amp[0], amp[2])
    res +=    -0.500000 * einsum('Ikc,cj,baki,I->abij', cc_obj.h1e1p.ov, amp[0], amp[1], amp[1])
    return res

def get_res_bra_2(cc_obj, amp):
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('J,IJ->I'           , cc_obj.h1p_eff, amp[3])
    res +=     1.000000 * einsum('IJ,J->I'           , cc_obj.h1p, amp[1])
    res +=     1.000000 * einsum('ia,Iai->I'         , cc_obj.h1e.ov, amp[2])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1e1p.ov, amp[0])
    res +=     1.000000 * einsum('Jia,IJai->I'       , cc_obj.h1e1p.ov, amp[4])
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , cc_obj.h1e1p.ov, amp[1], amp[2])
    res +=     1.000000 * einsum('Jia,ai,IJ->I'      , cc_obj.h1e1p.ov, amp[0], amp[3])
    return res

def get_res_bra_3(cc_obj, amp):
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1e1p.vo)
    res +=     1.000000 * einsum('J,IJai->Iai'       , cc_obj.h1p_eff, amp[4])
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , cc_obj.h1e.oo, amp[2])
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , cc_obj.h1e.vv, amp[2])
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , cc_obj.h1p, amp[2])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1e1p.oo, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1e1p.vv, amp[0])
    res +=     1.000000 * einsum('Jai,IJ->Iai'       , cc_obj.h1e1p.vo, amp[3])
    res +=    -1.000000 * einsum('Ijb,abji->Iai'     , cc_obj.h1e1p.ov, amp[1])
    res +=    -1.000000 * einsum('Jji,IJaj->Iai'     , cc_obj.h1e1p.oo, amp[4])
    res +=     1.000000 * einsum('Jab,IJbi->Iai'     , cc_obj.h1e1p.vv, amp[4])
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , cc_obj.h1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , cc_obj.h1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1e1p.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , cc_obj.h1e1p.oo, amp[1], amp[2])
    res +=    -1.000000 * einsum('Jji,aj,IJ->Iai'    , cc_obj.h1e1p.oo, amp[0], amp[3])
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , cc_obj.h1e1p.vv, amp[1], amp[2])
    res +=     1.000000 * einsum('Jab,bi,IJ->Iai'    , cc_obj.h1e1p.vv, amp[0], amp[3])
    res +=    -1.000000 * einsum('Jjb,aj,IJbi->Iai'  , cc_obj.h1e1p.ov, amp[0], amp[4])
    res +=    -1.000000 * einsum('Jjb,bi,IJaj->Iai'  , cc_obj.h1e1p.ov, amp[0], amp[4])
    res +=     1.000000 * einsum('Jjb,bj,IJai->Iai'  , cc_obj.h1e1p.ov, amp[0], amp[4])
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , cc_obj.h1e1p.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , cc_obj.h1e1p.ov, amp[2], amp[2])
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , cc_obj.h1e1p.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('Jjb,abji,IJ->Iai'  , cc_obj.h1e1p.ov, amp[1], amp[3])
    res +=    -0.500000 * einsum('Jjb,aj,J,Ibi->Iai' , cc_obj.h1e1p.ov, amp[0], amp[1], amp[2])
    res +=    -0.500000 * einsum('Jjb,bi,J,Iaj->Iai' , cc_obj.h1e1p.ov, amp[0], amp[1], amp[2])
    res +=    -0.500000 * einsum('Jjb,bi,aj,IJ->Iai' , cc_obj.h1e1p.ov, amp[0], amp[0], amp[3])
    return res

def get_res_bra_4(cc_obj, amp):
    res  =     1.000000 * einsum('IK,JK->IJ'         , cc_obj.h1p, amp[3])
    res +=     1.000000 * einsum('JK,IK->IJ'         , cc_obj.h1p, amp[3])
    res +=     1.000000 * einsum('ia,IJai->IJ'       , cc_obj.h1e.ov, amp[4])
    res +=     1.000000 * einsum('Iia,Jai->IJ'       , cc_obj.h1e1p.ov, amp[2])
    res +=     1.000000 * einsum('Jia,Iai->IJ'       , cc_obj.h1e1p.ov, amp[2])
    res +=     1.000000 * einsum('Kia,K,IJai->IJ'    , cc_obj.h1e1p.ov, amp[1], amp[4])
    res +=     1.000000 * einsum('Kia,Iai,JK->IJ'    , cc_obj.h1e1p.ov, amp[2], amp[3])
    res +=     1.000000 * einsum('Kia,Jai,IK->IJ'    , cc_obj.h1e1p.ov, amp[2], amp[3])
    return res

def get_res_bra_5(cc_obj, amp):
    res  =    -1.000000 * einsum('ji,IJaj->IJai'     , cc_obj.h1e.oo, amp[4])
    res +=     1.000000 * einsum('ab,IJbi->IJai'     , cc_obj.h1e.vv, amp[4])
    res +=     1.000000 * einsum('IK,JKai->IJai'     , cc_obj.h1p, amp[4])
    res +=     1.000000 * einsum('JK,IKai->IJai'     , cc_obj.h1p, amp[4])
    res +=    -1.000000 * einsum('Iji,Jaj->IJai'     , cc_obj.h1e1p.oo, amp[2])
    res +=     1.000000 * einsum('Iab,Jbi->IJai'     , cc_obj.h1e1p.vv, amp[2])
    res +=    -1.000000 * einsum('Jji,Iaj->IJai'     , cc_obj.h1e1p.oo, amp[2])
    res +=     1.000000 * einsum('Jab,Ibi->IJai'     , cc_obj.h1e1p.vv, amp[2])
    res +=    -1.000000 * einsum('jb,aj,IJbi->IJai'  , cc_obj.h1e.ov, amp[0], amp[4])
    res +=    -1.000000 * einsum('jb,bi,IJaj->IJai'  , cc_obj.h1e.ov, amp[0], amp[4])
    res +=    -1.000000 * einsum('jb,Iaj,Jbi->IJai'  , cc_obj.h1e.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('jb,Ibi,Jaj->IJai'  , cc_obj.h1e.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('Ijb,aj,Jbi->IJai'  , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,Jaj->IJai'  , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Jjb,aj,Ibi->IJai'  , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Jjb,bi,Iaj->IJai'  , cc_obj.h1e1p.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Kji,K,IJaj->IJai'  , cc_obj.h1e1p.oo, amp[1], amp[4])
    res +=    -1.000000 * einsum('Kji,Iaj,JK->IJai'  , cc_obj.h1e1p.oo, amp[2], amp[3])
    res +=    -1.000000 * einsum('Kji,Jaj,IK->IJai'  , cc_obj.h1e1p.oo, amp[2], amp[3])
    res +=     1.000000 * einsum('Kab,K,IJbi->IJai'  , cc_obj.h1e1p.vv, amp[1], amp[4])
    res +=     1.000000 * einsum('Kab,Ibi,JK->IJai'  , cc_obj.h1e1p.vv, amp[2], amp[3])
    res +=     1.000000 * einsum('Kab,Jbi,IK->IJai'  , cc_obj.h1e1p.vv, amp[2], amp[3])
    res +=    -1.000000 * einsum('Kjb,Iaj,JKbi->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -1.000000 * einsum('Kjb,Ibi,JKaj->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=     1.000000 * einsum('Kjb,Ibj,JKai->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -1.000000 * einsum('Kjb,Jaj,IKbi->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -1.000000 * einsum('Kjb,Jbi,IKaj->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=     1.000000 * einsum('Kjb,Jbj,IKai->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=     1.000000 * einsum('Kjb,Kai,IJbj->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -1.000000 * einsum('Kjb,Kaj,IJbi->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -1.000000 * einsum('Kjb,Kbi,IJaj->IJai', cc_obj.h1e1p.ov, amp[2], amp[4])
    res +=    -0.500000 * einsum('Kjb,K,Iaj,Jbi->IJai', cc_obj.h1e1p.ov, amp[1], amp[2], amp[2])
    res +=    -0.500000 * einsum('Kjb,K,Ibi,Jaj->IJai', cc_obj.h1e1p.ov, amp[1], amp[2], amp[2])
    res +=    -0.500000 * einsum('Kjb,aj,K,IJbi->IJai', cc_obj.h1e1p.ov, amp[0], amp[1], amp[4])
    res +=    -0.500000 * einsum('Kjb,aj,Ibi,JK->IJai', cc_obj.h1e1p.ov, amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('Kjb,aj,Jbi,IK->IJai', cc_obj.h1e1p.ov, amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('Kjb,bi,K,IJaj->IJai', cc_obj.h1e1p.ov, amp[0], amp[1], amp[4])
    res +=    -0.500000 * einsum('Kjb,bi,Iaj,JK->IJai', cc_obj.h1e1p.ov, amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('Kjb,bi,Jaj,IK->IJai', cc_obj.h1e1p.ov, amp[0], amp[2], amp[3])
    return res
