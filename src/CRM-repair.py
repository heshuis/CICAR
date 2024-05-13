import pandas as pd
import numpy as np
import re
import ast
import yaml
import sys


# return true if a positive causal rule can be applied to the case, so the case does not (fully) satisfy the LHS and LHS can be completely changed, so the variables needing change are controllable. 
def evaluatepositivecausalrule(LHS, case, cvl = None )-> bool:
    applicable = False
    for I in LHS:
        if case[I]!=1:
            applicable=True # at least one feature in the LHS should not be in the case
            if cvl!=None and I not in cvl:
                return False # LHS cannot be reached
    return applicable # False if LHS is already fully satisfied in case



# return true if a negative causal rule applies to the case, so the case satisfies the LHS and the LHS can be completely changed (is fully controllable)
def evaluatenegativecausalrule(LHS, case, cvl = None)-> bool:
    applicable=False
    for I in LHS:
        if case[I]!=1:
            return False
        if cvl!=None and I in cvl:
            applicable=True
    if cvl==None:
        applicable = True
    return applicable #at least one I in LHS should be changeable

def rulessatisfied(case,pRc,nRc):
    pL=set()
    for LHS in pRc:
        satisfied=True
        for I in LHS:
            if case[I]!=1:
                satisfied=False
                break
        if (satisfied):
            pL.add(tuple(LHS))
    nL=set()
    for LHS in nRc:
        if evaluatenegativecausalrule(LHS,case):
            nL.add(tuple(LHS))
    return pL,nL


def newcase(case,Fold,Fnew):
    case_new=case.copy()
    for p in Fold:
        if case_new[p]==1:
            case_new[p]=0
        else:
            case_new[p]=1
    for p in Fnew:
        if case_new[p]==1:
            case_new[p]=0
        else:
            case_new[p]=1
    return case_new

def isfullycontrollable(LHS,controllable_variables):
    for I in LHS:
        if I not in controllable_variables:
            return False
    return True

def ismixedcontrollable(LHS,controllable_variables):
    # true if LHS contains at least one controllable and one noncontrollable variable
    foundcontrollable=False
    foundnoncontrollable=False
    for I in LHS:
        if I not in controllable_variables:
            foundnoncontrollable=True
        if I in controllable_variables:
            foundcontrollable=True
    return (foundcontrollable and foundnoncontrollable)

def iscontrollable(LHS, controllable_variables):
    return isfullycontrollable(LHS,controllable_variables) or ismixedcontrollable(LHS,controllable_variables)

def exclusive(f,controllable_feature_list):
    l=[]
    for cfl in controllable_feature_list:
        if f in cfl:
            l = cfl[:]
            l.remove(f)
            return l
    return l

def areexclusive(f1,f2,controllable_feature_list):
    if f1==f2:
        return False
    for cfl in controllable_feature_list:
        if f1 in cfl and f2 in cfl:
            return True
    return False

def distance(f1,f2,controllable_feature_list):
    if originalvar[f1] in nominal_variables: # originalvar[f1]==originalvar[f2] because of the one-hot encoding
        return 1
    for cfl in controllable_feature_list:
        if f1 in cfl and f2 in cfl:
            return abs(cfl.index(f1)-cfl.index(f2))
    return -1



#alternative for negative feature f is returned.
def alternative(f,controllable_feature_list):
    E=exclusive(f,controllable_feature_list)
    lowestdist=len(E) # distance
    closestf=None
    for fn in E:
        d=distance(f,fn,controllable_feature_list)
        if d<=lowestdist:
            lowestdist=d
            closestf=fn
    return closestf,lowestdist

#alternative for negative feature f is returned. The alternative cannot be in Fold.
def limitedalternative(f,controllable_feature_list,Fold):
    E=exclusive(f,controllable_feature_list)
    lowestdist=len(E) # distance
    closestf=None
    for fn in E:
        if fn in Fold:
            continue;
        d=distance(f,fn,controllable_feature_list)
        if d<=lowestdist:
            lowestdist=d
            closestf=fn
    return closestf,lowestdist


# returns unique feature from the case that is alternative to f
def prevalternative(f,case,controllable_feature_list):
    E=exclusive(f,controllable_feature_list)
    for e in E:
        if case[e]==1:
            return e, distance(e,f,controllable_feature_list)
    return None, None

# selects feature in negative rule nLHS that is changed
# such that the closest alternative feature occurs most frequently in positive rule
def selectfeature(case,nLHS,cvl,pRc,nRc,cfl):
    prio= dict()
    alts= dict()
    altdist= dict()
    for p in nLHS:
        if case[p]==1 and p in cvl:
            alternatives=exclusive(p,cfl)
            usedalternatives=[item for rule in nRc for item in rule]
            viablealternatives=list (set(alternatives)-set(usedalternatives))
            if not viablealternatives: # empty list
                viablealternatives=usedalternatives # no unused featured
            lowestdist=len(alternatives)
            closestf=None
            for fn in viablealternatives:
                d=distance(p,fn,cfl)
                if d<=lowestdist:
                    lowestdist=d
                    closestf=fn
            if not closestf:
                print ('Error: no closest alternative found for feature ',p)
                exit()
            alts[p]=closestf
            altdist[p]=lowestdist
            prio[p]=sum (r.count(closestf) for r in pRc)
    if not prio:
        print ('Error: no alternative found for case ',case)
        print ('when applying negative rule with LHS ',nLHS)
        print ('cvl ',cvl)
        print(pRc)
        print(nRc)
        print(altdist)
        exit()
    mindistance=min(altdist.values()) # least change distance
    minkeys=[key for key in altdist if altdist[key]==mindistance] # all features with least change distance
    priokey= dict()
    for key in minkeys:
        priokey[key]=prio[key]    
    selected=max(priokey,key=priokey.get) # feature whose alternative occurs in most positive rule
    return selected,alts[selected],altdist[selected]

def conflict(LHS1,LHS2,controllable_feature_list):
    for l1 in LHS1:
        for l2 in LHS2:
            if areexclusive(l1,l2,controllable_feature_list):
                return True
    return False


def consistentRuleSets(LHSs,controllable_feature_list):
    ruleSets=[]
    conf=False
    for i in range(len(LHSs)-1):
        for j in range(i+1, len(LHSs)):
            if conflict(LHSs[i],LHSs[j],controllable_feature_list):
                conf=True
                # construct two alternative rule sets: one without LHSs[i], one without LHSs[j]
                if (i>0):
                    ruleSets.extend(consistentRuleSets(LHSs[0:i]+LHSs[i+1:len(LHSs)],controllable_feature_list))
                else:
                    ruleSets.extend(consistentRuleSets(LHSs[i+1:len(LHSs)],controllable_feature_list))
                if (j==len(LHSs)):
                    ruleSets.extend(consistentRuleSets(LHSs[0:len(LHSs)-1],controllable_feature_list))
                else:
                    ruleSets.extend(consistentRuleSets(LHSs[0:j]+LHSs[j+1:len(LHSs)],controllable_feature_list))
    if not conf and LHSs!=[]:
        ruleSets=[LHSs]
    return ruleSets

def maxRuleSets(LHSs,controllable_feature_list):
    cRs=consistentRuleSets(LHSs,controllable_feature_list)
    mRs = [] # set of maximal rule sets
    # remove duplicates
    for r in cRs: 
        if r not in mRs:
            mRs.append(r)
    nonMaximal=[]
    # compute non-maximal rule sets 
    for r1 in mRs:
        for r2 in mRs:
            if r1==r2: continue
            subset=True
            for i in r1:
                if i not in r2:
                    subset=False
                    break
            if subset:
                nonMaximal.append(r1)  
    # return maximal rule sets
    l=[r for r in mRs if r not in nonMaximal]
    if len(l)==1 and len(cRs)>1: 
        print("not enough multiple consistent sets")
        print(l)
    return l

# remove conflicting features from Fpos. The one closest to the existing feature f in Fprev is kept.
def resolveconflicts(Fpos,Fprev):
    toremove=[]
    for f1 in Fpos:
        for f2 in Fpos:
            if areexclusive(f1,f2):
                for f in Fprev:
                    if areexclusive(f,f1):
                        if distance(f,f1)<distance(f,f2):
                            toremove.append(f2)
                        else:
                            toremove.append(f1)
    Fposnew= [ f for f in Fpos if (f not in toremove) ]
    return Fposnew

def maxSets(case,nLHSs,cvl):
    maxS=list()
    if len(nLHSs)==1:
        LHS=nLHSs.pop()
        for p in LHS:
            if case[p]==1 and p in cvl:
                maxS.append([p])
        return maxS
    if (len(nLHSs)>1): 
        LHS=nLHSs.pop()
        maxS=maxSets(case,nLHSs,cvl)
        nmS=[]
        for m in maxS:
            for p in LHS:
                mp=list(m)
                if case[p]==1 and p in cvl:
                    if p not in mp:
                        mp.append(p)
                        nmS.append(mp)
        return nmS
    return []

def maxSetsHeuristics(case,nLHSs,cvl):
    occurrences= dict()
    for LHS in nLHSs:
        for p in LHS:
            if case[p]==1 and p in cvl:
                if p not in occurrences:
                    occurrences[p]=1
                else:
                    occurrences[p]+=1
    candidates=dict(sorted(occurrences.items(),key=lambda x:x[1], reverse=True))
    return candidates


def findoutcomesalternativecase(dataset,case,relevant_variables):
    mask=1
    for v in relevant_variables:
        if case[v]==1:
            mask=mask & (dataset[v]==1)
        else:
            mask=mask & (data[v]==0)
    return (dataset[mask]['Z'].count(), dataset[mask]['Z'].sum(), (dataset[mask]['Z']==0).sum() )

def findoutcomealternativecase(dataset,case,new_selected_variables, Fold,Fnew):
    mask=1
    for v in new_selected_variables:
        if v in Fold or v in Fnew:
            continue
        if case[v]==1:
            mask=mask & (dataset[v]==1)
        else:
            mask=mask & (data[v]==0)
    for f in Fold:
        mask=mask & (dataset[f]==0)
    for f in Fnew:
        mask=mask & (dataset[f]==1)
    alternativeoutcomes=dataset[mask]
    return (dataset[mask]['Z'].count(), dataset[mask]['Z'].sum(), (dataset[mask]['Z']==0).sum() )

def findoutcomealternativecase1(dataset,case,Fneg,Falt,Fprev,Fpos):
    mask=1
    for v in new_selected_variables:
        if v in Fneg or v in Falt or v in Fprev or v in Fpos:
            continue
        if case[v]==1:
            mask=mask & (dataset[v]==1)
        else:
            mask=mask & (data[v]==0)
    for f in Fneg:
        mask=mask & (dataset[f]==0)
    for f in Falt:
        mask=mask & (dataset[f]==1)
    for f in Fprev:
        mask=mask & (dataset[f]==0)
    for f in Fpos:
        mask=mask & (dataset[f]==1)
    alternativeoutcomes=dataset[mask]
    return (dataset[mask]['Z'].count(), dataset[mask]['Z'].sum(), (dataset[mask]['Z']==0).sum() )


def selectpositiverule(pRc,case, prio: str,cfl):
    if prio=="odds":
        oddsr=0
        mindistance=100
        selected=[]
        for LHS in pRc:
            if odds[tuple(LHS)]>oddsr:
                oddsr=odds[tuple(LHS)]
                selected=LHS
        if not selected:
            print ('no rule selected')
            print (case)
            print (pRc)
        return selected
    if prio=="changedistance":
        mindistance=100
        oddsr=0
        selected=[]
        for LHS in pRc:
            changedistance=0
            for p in LHS:
                if case[p]==0:
                    fold,dist=prevalternative(p,case,cfl)
                    if fold==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        changedistance+=dist
            if changedistance<mindistance:
                mindistance=changedistance
                oddsr=odds[tuple(LHS)]
                selected=LHS
            else: 
                if changedistance==mindistance:
                   if odds[tuple(LHS)]>oddsr:
                        oddsr=odds[tuple(LHS)]
                        selected=LHS 
        return selected

def selectnegativerule(nRc):
    oddsr=0
    selected=[]
    for LHS in nRc:
        if odds[tuple(LHS)]>oddsr:
            oddsr=odds[tuple(LHS)]
            selected=LHS
    return selected
            
def calcaggregateoddsratio(mRS):
    aggoddsr=0
    for LHS in mRS:
        aggoddsr+=odds[tuple(LHS)]
    return aggoddsr

def selectpositiveruleset(mRSs,case,prio:str,cfl):
    if prio=="odds":
        maxaggoddsr=0
        selected=[]
        for mRS in mRSs:
            aggoddsr=calcaggregateoddsratio(mRS)
            if aggoddsr>maxaggoddsr:
                maxaggoddsr=aggoddsr
                selected=mRS
        return selected
    if prio=="changedistance":
        minaggdistance=100
        maxaggoddsr=0
        selected=[]
        for mRS in mRSs:
            for LHS in mRS:
                changedistance=0
                for p in LHS:
                    if case[p]==0:
                        fold,dist=prevalternative(p,case,cfl)
                        if fold==None:
                            print ('Error: no alternative feature found for ', p)
                            error=True
                            exit()
                        else:
                            changedistance+=dist
            if changedistance < minaggdistance:
                minaggdistance=changedistance
                maxaggoddsr=calcaggregateoddsratio(mRS)
                selected=mRS
            else:
                if changedistance == minaggdistance:
                    aggoddsr = calcaggregateoddsratio(mRS)
                    if aggoddsr>maxaggoddsr:
                        maxaggoddsr=aggoddsr
                        selected=mRS
        return selected

def selectnegativeruleset(mRSs):
    maxaggoddsr=0
    selected=[]
    for mRS in mRSs:
        aggoddsr=calcaggregateoddsratio(mRS)
        if aggoddsr>maxaggoddsr:
            maxaggoddsr=aggoddsr
            selected=mRS
    return selected

def generate_final_results_alt(pRc, nRc, relevant_variables, cvl, cfl):
    k=0
    df_cases = pd.DataFrame(columns = ['Index', 'Satisfied positive rules before repair', 'Satisfied negative rules before repair', 'Strategy', 'Rules', 'Intervention size','Fold','Fnew','Reach','PReach','NReach','Success ratio before repair', 'Success ratio after repair','Success ratio improvement', 'Satisfied positive rules after repair','Satisfied negative rules after repair',  'Change distance'])
    for index, case in data.iterrows():
        error=False
        if (case['Z']==1): continue # only process fail cases

        k+=1
        relevantpRc=[]
        relevantnRc=[]

        for LHS in pRc:
            if iscontrollable(LHS,cvl) and evaluatepositivecausalrule(LHS,case,cvl):
                relevantpRc.append(LHS)
        for LHS in nRc:
            if iscontrollable(LHS,cvl) and evaluatenegativecausalrule(LHS,case,cvl):
                relevantnRc.append(LHS)

        print ('Case ', k, ': ', case)
        for r in relevantpRc:
            print(r , '------> Z is applicable' )
        for r in relevantnRc:
            print(r , '------> !Z is applicable')

        pL,nL=rulessatisfied(case,pRc,nRc)               

        (alt_cases_old, pos_alt_old, neg_alt_old)= findoutcomesalternativecase(data,case, relevant_variables)

        #strategy 1: single pos
        for LHS in relevantpRc:
            changedistance=0
            Fold=[]
            Fnew=[]
            for p in LHS:
                if case[p]==0:
                    fold,dist=prevalternative(p,case,cfl)
                    if fold==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        changedistance+=dist
                        Fnew.append(p)
                        Fold.append(fold)
            if not error: #apply full changes to enable r
                case_new=newcase(case,Fold,Fnew) 
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc)  
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))
                df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL), 'Strategy':1, 'Rules':[LHS], 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new), 'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),   'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),  'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance},
                                       ignore_index = True)
        #strategy 1a
        LHS=selectpositiverule(relevantpRc,case,"odds",cfl)
        changedistance=0
        Fold=[]
        Fnew=[]
        for p in LHS:
            if case[p]==0:
                fold,dist=prevalternative(p,case,cfl)
                if fold==None:
                    print ('Error: no alternative feature found for ', p)
                    error=True
                    exit()
                else:
                    changedistance+=dist
                    Fnew.append(p)
                    Fold.append(fold)
        if not error and len(Fold)>0: #apply full changes to enable r
            case_new=newcase(case,Fold,Fnew) 
            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc)  
            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
            improvement_ratio = improvement / (len(pRc)+len(nRc))
            df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL), 'Strategy':'1a', 'Rules':[LHS], 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new), 'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),   'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance},
                                       ignore_index = True)
        #strategy 1b
        LHS=selectpositiverule(relevantpRc,case,"changedistance",cfl)
        changedistance=0
        Fold=[]
        Fnew=[]
        for p in LHS:
            if case[p]==0:
                fold,dist=prevalternative(p,case,cfl)
                if fold==None:
                    print ('Error: no alternative feature found for ', p)
                    error=True
                    exit()
                else:
                    changedistance+=dist
                    Fnew.append(p)
                    Fold.append(fold)
        if not error and len(Fold)>0: #apply full changes to enable r
            case_new=newcase(case,Fold,Fnew) 
            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc)  
            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
            improvement_ratio = improvement / (len(pRc)+len(nRc))
            df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL), 'Strategy':'1b', 'Rules':[LHS], 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new), 'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),   'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance},
                                       ignore_index = True)
        #strategy 2: single neg
        for LHS in relevantnRc:
            for p in LHS:
                changedistance=0
                Fold=[]
                Fnew=[]
                if case[p]==1 and p in cvl:
                    fnew,dist=alternative(p,cfl)
                    if fnew==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        changedistance+=dist
                        Fold.append(p)
                        Fnew.append(fnew)
                    if not error: #apply one change to disable r
                        case_new=newcase(case,Fold,Fnew)  
                        (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                        pLnew,nLnew=rulessatisfied(case_new,pRc,nRc)
                        improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                        improvement_ratio = improvement / (len(pRc)+len(nRc))  
                        df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL), 'Strategy':2, 'Rules':[LHS], 'Intervention size': 1,'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        #strategy 2a: single neg
        LHS=selectnegativerule(relevantnRc)
        if not LHS:
            print ('no relevant negative rule applicable')
            print (relevantnRc)
            if relevantnRc: 
                exit()
        else:
            p,f,changedistance=selectfeature(case,LHS,cvl,pRc,nRc,cfl)
        
            Fold=[p]
            Fnew=[f]
        
            case_new=newcase(case,Fold,Fnew)  
            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc)
            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
            improvement_ratio = improvement / (len(pRc)+len(nRc))  
            df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL), 'Strategy':'2a', 'Rules':[LHS], 'Intervention size': 1,'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        #strategy 3: all pos
        mRSs=maxRuleSets(relevantpRc,cfl)
        for mRS in mRSs:
            Fold=[]
            Fnew=[]
            changedistance=0
            for LHS in mRS:
                for p in LHS:
                    if case[p]==0:
                        fold,dist=prevalternative(p,case,cfl)
                        if fold==None:
                            print ('Error: no alternative feature found for ', p)
                            error=True
                            exit()
                        else:
                            changedistance+=dist
                            Fnew.append(p)
                            Fold.append(fold)
            if not error:
                case_new=newcase(case,Fold,Fnew)  
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))  
                df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':3, 'Rules':mRS, 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old), 'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        #strategy 3a
        mRS=selectpositiveruleset(mRSs,case,"odds",cfl)
        Fold=[]
        Fnew=[]
        changedistance=0
        for LHS in mRS:
            for p in LHS:
                    if case[p]==0:
                        fold,dist=prevalternative(p,case,cfl)
                        if fold==None:
                            print ('Error: no alternative feature found for ', p)
                            error=True
                            exit()
                        else:
                            changedistance+=dist
                            Fnew.append(p)
                            Fold.append(fold)
        if not error:
            case_new=newcase(case,Fold,Fnew)  
            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
            improvement_ratio = improvement / (len(pRc)+len(nRc))  
            df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'3a', 'Rules':mRS, 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old), 'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        # strategy 3b
        mRS=selectpositiveruleset(mRSs,case,"changedistance",cfl)   
        Fold=[]
        Fnew=[]
        changedistance=0
        for LHS in mRS:
            for p in LHS:
                    if case[p]==0:
                        fold,dist=prevalternative(p,case,cfl)
                        if fold==None:
                            print ('Error: no alternative feature found for ', p)
                            error=True
                            exit()
                        else:
                            changedistance+=dist
                            Fnew.append(p)
                            Fold.append(fold)
        if not error:
            case_new=newcase(case,Fold,Fnew)  
            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
            improvement_ratio = improvement / (len(pRc)+len(nRc))  
            df_cases = df_cases.append({'Index' : index, 'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'3b', 'Rules':mRS, 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old), 'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        # strategy 4
        mRSs=maxSets(case,relevantnRc.copy(),cvl)
        for mRS in mRSs: # mRS is a set of vars, each occurring in one LHS of the applicable negative rules 
            Fold=mRS
            Fnew=[]
            changedistance=0
            for p in mRS:
                if case[p]==1 and p in cvl:
                    fnew,dist=limitedalternative(p,cfl,Fold)
                    if fnew==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        changedistance+=dist
                        Fnew.append(fnew)
            if not error:
                case_new=newcase(case,Fold,Fnew)
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))  
                df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':4, 'Rules':mRS, 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
                
        # strategy 4a
        mRS=maxSetsHeuristics(case,relevantnRc.copy(),cvl) #mRs contains a sorted list of vars occurring in the LHS of rules in relevantnRc
        selected=[]
        tobecovered=relevantnRc.copy()
        done=False
        for p in mRS:
            selected.append(p)
            for r in relevantnRc:
                if p in r and r in tobecovered:
                    tobecovered.remove(r)
            if tobecovered==[]:
                done=True
                break
        if (done):
            Fold=selected
            Fnew=[]
            changedistance=0
            for p in selected:
                if case[p]==1 and p in cvl:
                    fnew,dist=limitedalternative(p,cfl,Fold)
                    if fnew==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        changedistance+=dist
                        Fnew.append(fnew)
            if not error:
                case_new=newcase(case,Fold,Fnew)
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))  
                df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'4a', 'Rules':mRS, 'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        
        #strategy 5
        mRSs=maxRuleSets(relevantpRc,cfl)
        for mRS in mRSs:
            Fold=[]
            Fnew=[]
            changedistance=0

            for LHS in mRS:
                for p in LHS:
                    if case[p]==0:
                        fold,dist=prevalternative(p,case,cfl)
                        changedistance+=dist
                        if fold==None:
                            print ('Error: no alternative feature found for ', p)
                            error=True
                            exit()
                        else:
                            Fnew.append(p)
                            Fold.append(fold)
            if not error:                
                relevantnRc_new=[]
                for LHS in relevantnRc:
                    found=False
                    for p in LHS:
                        if p in Fold: #already p is modified, so LHS not be true
                            found=True
                            break
                    if not found:
                        relevantnRc_new.append(LHS)
                if relevantnRc_new:
                    mRSsneg=maxSets(case,relevantnRc_new.copy(),cvl)
                    for mRSn in mRSsneg:
                        Foldnew=Fold.copy()
                        Foldnew.extend(mRSn)
                        for p in mRSn:
                            if case[p]==1 and p in cvl:
                                fnew,dist=limitedalternative(p,cfl,Foldnew)
                                if fnew==None:
                                    print ('Error: no alternative feature found for ', p)
                                    error=True
                                    exit()
                                else:
                                    changedistance+=dist
                                    Fnew.append(fnew)
                        if not error:
                            case_new=newcase(case,Foldnew,Fnew)
                            (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                            pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                            improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                            improvement_ratio = improvement / (len(pRc)+len(nRc))  
                            df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':5, 'Rules':mRSsneg, 'Intervention size': len(Foldnew),'Fold' : Foldnew, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
                else:
                    case_new=newcase(case,Fold,Fnew)
                    (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                    pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                    improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                    improvement_ratio = improvement / (len(pRc)+len(nRc))  
                    df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':5,'Rules':mRS,  'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        # strategy 5a
        mRS=selectpositiveruleset(mRSs,case,"odds",cfl)   
        Fold=[]
        Fnew=[]
        changedistance=0

        for LHS in mRS:
            for p in LHS:
                if case[p]==0:
                    fold,dist=prevalternative(p,case,cfl)
                    changedistance+=dist
                    if fold==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        Fnew.append(p)
                        Fold.append(fold)
        if not error:
            relevantnRc_new=[]
            for LHS in relevantnRc:
                found=False
                for p in LHS:
                    if p in Fold: #already p is modified, so LHS not be true
                        found=True
                        break
                if not found:
                    relevantnRc_new.append(LHS)
            if relevantnRc_new:
                # similar to strategy 4a
                mRSn=maxSetsHeuristics(case,relevantnRc_new.copy(),cvl) #mRs contains a sorted list of vars occurring in the LHS of rules in relevantnRc
                selected=[]
                tobecovered=relevantnRc.copy()
                done=False
                for p in mRS:
                    selected.append(p)
                    for r in relevantnRc:
                        if p in r and r in tobecovered:
                            tobecovered.remove(r)
                    if tobecovered==[]:
                        done=True
                        break
                if (done):
                    Foldnew=Fold.copy()
                    Foldnew.extend(selected)
                    changedistance=0
                    for p in selected:
                        if case[p]==1 and p in cvl:
                            fnew,dist=limitedalternative(p,cfl,Foldnew)
                            if fnew==None:
                                print ('Error: no alternative feature found for ', p)
                                error=True
                                exit()
                            else:
                                changedistance+=dist
                                Fnew.append(fnew)
                    if not error:
                        case_new=newcase(case,Foldnew,Fnew)
                        (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                        pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                        improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                        improvement_ratio = improvement / (len(pRc)+len(nRc))  
                        df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'5a', 'Rules':mRSsneg, 'Intervention size': len(Foldnew),'Fold' : Foldnew, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
            else:
                case_new=newcase(case,Fold,Fnew)
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))  
                df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'5a','Rules':mRS,  'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew), 'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
        # strategy 5b
        mRS=selectpositiveruleset(mRSs,case,"changedistance",cfl)   
        Fold=[]
        Fnew=[]
        changedistance=0

        for LHS in mRS:
            for p in LHS:
                if case[p]==0:
                    fold,dist=prevalternative(p,case,cfl)
                    changedistance+=dist
                    if fold==None:
                        print ('Error: no alternative feature found for ', p)
                        error=True
                        exit()
                    else:
                        Fnew.append(p)
                        Fold.append(fold)
        if not error:
            relevantnRc_new=[]
            for LHS in relevantnRc:
                found=False
                for p in LHS:
                    if p in Fold: #already p is modified, so LHS not be true
                        found=True
                        break
                if not found:
                    relevantnRc_new.append(LHS)
            if relevantnRc_new:
                # similar to strategy 4a
                mRSn=maxSetsHeuristics(case,relevantnRc_new.copy(),cvl) #mRs contains a sorted list of vars occurring in the LHS of rules in relevantnRc
                selected=[]
                tobecovered=relevantnRc.copy()
                done=False
                for p in mRS:
                    selected.append(p)
                    for r in relevantnRc:
                        if p in r and r in tobecovered:
                            tobecovered.remove(r)
                    if tobecovered==[]:
                        done=True
                        break
                if (done):
                    Foldnew=Fold.copy()
                    Foldnew.extend(selected)
                    changedistance=0
                    for p in selected:
                        if case[p]==1 and p in cvl:
                            fnew,dist=limitedalternative(p,cfl,Foldnew)
                            if fnew==None:
                                print ('Error: no alternative feature found for ', p)
                                error=True
                                exit()
                            else:
                                changedistance+=dist
                                Fnew.append(fnew)
                    if not error:
                        case_new=newcase(case,Foldnew,Fnew)
                        (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                        pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                        improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                        improvement_ratio = improvement / (len(pRc)+len(nRc))  
                        df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'5b', 'Rules':mRSsneg, 'Intervention size': len(Foldnew),'Fold' : Foldnew, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new,'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)
            else:
                case_new=newcase(case,Fold,Fnew)
                (alt_cases_new, pos_alt_new, neg_alt_new)= findoutcomesalternativecase(data,case_new, relevant_variables)
                pLnew,nLnew=rulessatisfied(case_new,pRc,nRc) 
                improvement = len(pLnew)-len(pL) + len(nL)-len(nLnew)
                improvement_ratio = improvement / (len(pRc)+len(nRc))  
                df_cases = df_cases.append({'Index' : index,  'Satisfied positive rules before repair':len(pL),'Satisfied negative rules before repair':len(nL),  'Strategy':'5b','Rules':mRS,  'Intervention size': len(Fold),'Fold' : Fold, 'Fnew' : Fnew ,   'Reach':alt_cases_new, 'PReach':pos_alt_new, 'NReach':neg_alt_new, 'Success ratio before repair': (pos_alt_old/alt_cases_old), 'Success ratio after repair': (pos_alt_new/alt_cases_new),'Success ratio improvement' :(pos_alt_new/alt_cases_new)-(pos_alt_old/alt_cases_old),  'Satisfied positive rules after repair':len(pLnew),'Satisfied negative rules after repair':len(nLnew),'Negative rule change': len(nLnew)-len(nL),'Rules improvement': improvement, 'Rules improvement ratio': improvement_ratio, 'Change distance': changedistance },
                                       ignore_index = True)        
   
    df_cases.to_csv("repair_results_"+name+".csv", sep=";", index=False)
    
# one hot encoding of multi-valued variable 'Rules', containing for each case the applied causal rules in repairing the case
def expand_cases(df_cases,df_rules):
    pRc=[]
    nRc=[]
    for index, rule in df_rules.iterrows():
        if rule['Controllable']==False: continue
        rulestr=str(rule['Rule'])
        if rulestr.endswith("--> Z"):
            rulestr=rulestr.replace("--> Z","")
            pRc.append(ast.literal_eval(rulestr.strip()))
        else:
            rulestr=rulestr.replace("--> !Z","")
            nRc.append(ast.literal_eval(rulestr.strip()))
    columnlist=[]
    for r in pRc:
        columnlist.append(str(r))
    for r in nRc:
        columnlist.append(str(r)) 
    df_rules2=pd.DataFrame(columns=columnlist)
    for r in pRc:
        df_rules2[str(r)]=[1 if str(r) in rs else 0 for rs in df_cases['Rules']]
    for r in nRc:
        df_rules2[str(r)]=[1 if str(r) in rs else 0 for rs in df_cases['Rules']]
    df=pd.concat([df_cases, df_rules2], axis="columns")
    return df,pRc,nRc

def analyze_stat(df_cases,pRc,nRc):              
    df_stat = pd.DataFrame(columns = ['Strategy', 'Min change', 'Max change', 'Mean change', 'Median change', 'Min Success ratio improvement', 'Max Success ratio improvement', 'Mean Success ratio improvement','Median Success ratio improvement', 'Min negative rule change', 'Max negative rule change', 'Mean negative rule change','Median negative rule change',  'Min reach', 'Max reach', 'Mean reach','Median reach', 'Unique outcomes'])
    for i in range(1,6):
        df_cases_sub=df_cases[df_cases['Strategy']==i]
        vcount=df_cases_sub['Index'].value_counts(sort=False)
        lcount=list(vcount)
        unique_outcomes=lcount.count(1)*100/len(vcount)
        df_stat = df_stat.append({'Strategy' : i, 'Min change':df_cases_sub['Change distance'].min(), 'Max change':df_cases_sub['Change distance'].max(), 'Mean change':df_cases_sub['Change distance'].mean(),'Median change':df_cases_sub['Change distance'].median(), 'Min Success ratio improvement':df_cases_sub['Success ratio improvement'].min(), 'Max Success ratio improvement':df_cases_sub['Success ratio improvement'].max(), 'Mean Success ratio improvement':df_cases_sub['Success ratio improvement'].mean(),'Median Success ratio improvement':df_cases_sub['Success ratio improvement'].median(),'Min negative rule change':df_cases_sub['Negative rule change'].min(), 'Max negative rule change':df_cases_sub['Negative rule change'].max(), 'Mean negative rule change':df_cases_sub['Negative rule change'].mean(),'Median negative rule change':df_cases_sub['Negative rule change'].median(), 'Min reach':df_cases_sub['Reach'].min(), 'Max reach':df_cases_sub['Reach'].max(), 'Mean reach':df_cases_sub['Reach'].mean(),'Median reach':df_cases_sub['Reach'].median(), 'Unique outcomes': unique_outcomes },
                                       ignore_index = True)
    df_stat.to_csv("repair_results_"+name+"-stats.csv",sep=";",index=False)
    df_stat_rules = pd.DataFrame(columns = ['Rule', 'Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4', 'Strategy 5'])
    for r in pRc:
        counts=[]
        for i in range(1,6):
            df_cases_sub=df_cases[df_cases['Strategy']==i]
            counts.append(df_cases_sub[str(r)].sum())
        df_stat_rules=df_stat_rules.append({'Rule' : str(r)+ "--> Z", 'Strategy 1':counts[0], 'Strategy 2':counts[1], 'Strategy 3':counts[2], 'Strategy 4':counts[3], 'Strategy 5':counts[4]}, ignore_index = True)
    for r in nRc:
        counts=[]
        for i in range(1,6):
            df_cases_sub=df_cases[df_cases['Strategy']==i]
            counts.append(df_cases_sub[str(r)].sum())
        df_stat_rules=df_stat_rules.append({'Rule' :  str(r)+ "--> Z", 'Strategy 1':counts[0], 'Strategy 2':counts[1], 'Strategy 3':counts[2], 'Strategy 4':counts[3], 'Strategy 5':counts[4]}, ignore_index = True)
    df_stat_rules.to_csv("repair_results_"+name+"_stats-rules.csv",sep=";",index=False)  



    

# from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def one_hot_encoding(input, column_name: str, output):
    """ For every unique value in the original column a seperate column will be added in the binary dataframe called 'data',
    and when the column only has 2 values, only 1 column will be created,
    where for both the name of the column is the name of the column of origin and the selected value combined,
    where for both 1 represents the value in the original column was equal to the selected value and 0 otherwise,
    the function returns for binary columns the encoder and for the other columns, the names of the original values.
    """
    
    # There is only 1 column needed to decifer the original value if the original column had only 2 values
    nv=[] # new variables
    ncv=[] # new controllable variables
    cfl=[] # controllable features, organised as list of lists. 
    
    if len(input[column_name].unique()) >= 2:
        new_cfl=[]
        values=df[column_name].astype(str).unique()
        sorted_values=sorted(values,key=natural_keys)
        for value in sorted_values:
            binary_list = []
            for i in input[column_name]:
                if str(i) == value:
                    binary_list.append(1)
                else:
                    binary_list.append(0)
            output[column_name +"_"+ value] = binary_list
            nv.append(column_name +"_"+ str(value))
            if column_name in controllable_variables:
                ncv.append(column_name +"_"+ value)
                new_cfl.append(column_name +"_"+ value)
        if new_cfl!=[]:
            cfl.append(new_cfl)
        return list(input[column_name].unique()), nv, ncv, cfl
    
def original(v):
    return originalvar[v] 


if len(sys.argv)==1:
    print ('yaml file argument missing')
    exit()
else:
    with open(sys.argv[1], "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        file = config['dataset_file']
        name = config['name']
        selected_variables=config['selected variables']
        controllable_variables=config['controllable variables']
        nominal_variables=config['nominal variables']
        target = config['target']

df =pd.read_csv(r"./%s" % file)
data = pd.DataFrame()
data['Z'] = df[target]
data['notZ'] = 1 - data['Z']
new_variables=[]
new_controllable_variables=[]
controllable_feature_list =[] # double array, consisting of lists of related (exlusive) features


# Execute the one-hot encoding and retrieve the encoding scheme
encoding_scheme = {}
originalvar= {} # links one-hot encoded var to original var

for i in selected_variables:
    encoding_scheme[i],nv,ncv,cfl = one_hot_encoding(df,i,data)
    for v in nv:
        originalvar[v]=i
    new_controllable_variables.extend(ncv)
    if (cfl!=[]):
        controllable_feature_list.extend(cfl)
#Set the variables to one-hot encoded column_names and pop the target variable
new_selected_variables = list(data.columns)
new_selected_variables.pop(0)
new_selected_variables.pop(0)


odds=dict()
df_rules=pd.read_csv("rules_"+name+".csv",sep=";")

pRc=[]
nRc=[]
for index, rule in df_rules.iterrows():
    if rule['Controllable']==False: continue
    rulestr=str(rule['Rule'])
    if rulestr.endswith("--> Z"):
        rulestr=rulestr.replace("--> Z","")
        r=ast.literal_eval(rulestr.strip())
        pRc.append(r)
        odds[tuple(r)]=rule['Odds ratio']
    else:
        rulestr=rulestr.replace("--> !Z","")
        r=ast.literal_eval(rulestr.strip())
        nRc.append(r)
        odds[tuple(r)]=rule['Odds ratio']

pRc_to_evalute=pRc
nRc_to_evalute=nRc
generate_final_results_alt(pRc_to_evalute, nRc_to_evalute, new_selected_variables, new_controllable_variables, controllable_feature_list)



