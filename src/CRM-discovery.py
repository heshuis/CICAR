import pandas as pd
import numpy as np
import math
import re
import yaml
import sys

# prefix tree
class Node:
    def __init__(self, itemName, itemSet, parentNodeList,d):
        self.itemName = itemName
        self.itemSet = itemSet
        self.lsupp = 0
        self.parentlist = parentNodeList
        self.children = set ()
        self.depth=d;


    def display(self, ind=1):
        print('  ' * ind, self.itemSet, ' ', self.lsupp)
        for child in list(self.children):
            child.display(ind+1)

def generateNextLevelTree(treeNode, root, level, dataset, RHS):
    if treeNode.depth <level-1: 
        for c in treeNode.children:
            generateNextLevelTree(c, root, level, dataset, RHS)
    else: 
        if treeNode.depth==level-1:
            for c1 in treeNode.children:
                for c2 in treeNode.children:
                    if c1==c2: continue
                    itemSet=c1.itemSet.union(c2.itemSet)
                    if (len(itemSet)==0):
                        print("non-empty itemset expected")
                        exit()
                    if contain(root,root,itemSet)!=None:
                        continue # similar node has been created already
                    stop = False
                    parentNodeList = list()
                    for v in itemSet:
                        itemSetminV =itemSet.copy()
                        itemSetminV.remove(v)
                        if len(itemSetminV)==0:
                             continue;
                        n = contain(root,root,itemSetminV)
                        if n== None:
                            stop = True
                            break
                        else:
                            parentNodeList.append(n)
                    if stop: continue
                    itemName= itemSet - c1.itemSet
                    itemName=itemName.pop()
                    if not c1 in parentNodeList:
                        parentNodeList.append(c1)
                    newNode = Node(itemName,itemSet,parentNodeList,c1.depth+1)
                    newNode.lsupp =local_support(itemSet,RHS, dataset)
                    c1.children.add(newNode)


def contain(treeNode, root, itemSet: set):
    if itemSet == set():
        # should be impossible
        treeNode.display(10)
        exit()
    if treeNode == root:
        for c in treeNode.children:
            n=contain(c,root,itemSet)
            if n!= None:
                return n
    else:
        if treeNode.itemSet == itemSet:
            return treeNode
        else: 
            if treeNode.itemName not in itemSet:
                return None
            for c in treeNode.children:
               n=contain(c,root,itemSet)
               if n!= None :
                  return n
    return None


def prune(T,minsup):
    if T.depth>0 and T.lsupp < minsup:
        for parent in T.parentlist:
            parentchildren=parent.children
            parentchildren.discard(T)
    else:
        for c in T.children.copy():
            prune(c,minsup)

#observation 2 from paper
def pruneSameSupport(T):
    if T.depth>0 :
        same=True
        for parent in T.parentlist:
            if T.lsupp!=parent.lsupp:
                same=False
                break
        if same:
            parentchildren=parent.children
            parentchildren.discard(T)
        else:
           for c in T.children.copy():
            pruneSameSupport(c) 
    else:
        for c in T.children.copy():
            pruneSameSupport(c)

def removeNode(T, itemSet):
    if T.itemSet==itemSet:
        for parent in T.parentlist:
            parentchildren=parent.children
            parentchildren.discard(T)
        exit
    else:
        for c in T.children.copy():
            removeNode(c,itemSet)

def generateAssocationRules(T,RHS,k,dataset):
    LHS_AR=[]
    if T.depth==k:
        if association_test(list(T.itemSet),RHS,dataset):
            LHS_AR.append(list(T.itemSet))
    else:
        for c in T.children:
            LHS_AR=LHS_AR+generateAssocationRules(c,RHS,k,dataset)
    return LHS_AR
  

def support_counter(var_list,RHS, dataset): 
    # with supp_p_RHS, all variables from the variable list and target RHS are 1. Thus creating a boolean mask to set all variables from 
    # the list equal to 1 and then sum the RHS values should give supp_p_RHS
    
    #Create for the variables in the list a mask where the values all should be equal to 1
    p = 1
    for v in var_list:
        p = p & (dataset[v] ==1)
    supp_p_RHS = dataset[p][RHS].sum()
    supp_p_not_RHS = dataset[p][RHS].count() - supp_p_RHS    
    not_p = ~p
    supp_not_p_RHS = dataset[not_p][RHS].sum()
    supp_not_p_not_RHS = dataset[not_p][RHS].count() - dataset[not_p][RHS].sum()
    return supp_p_RHS, supp_p_not_RHS, supp_not_p_not_RHS, supp_not_p_RHS


def local_support(var_list, RHS, dataset) -> float:
    """ Uses the support_counter() function to calculate the local support"""
    
    supp_p_RHS, supp_p_not_RHS, supp_not_p_not_RHS, supp_not_p_RHS = support_counter(var_list, RHS, dataset)
    supp_RHS = supp_p_RHS + supp_not_p_RHS
    l_supp = supp_p_RHS/supp_RHS
    
    return l_supp

# compute metrics support, confidence, lift of rule var_list -> RHS on dataset
def computeMetrics (var_list, RHS, dataset):
    supp_p_RHS, supp_p_not_RHS, supp_not_p_not_RH, supp_not_p_RHS = support_counter(var_list, RHS, dataset)
    supp = supp_p_RHS / dataset.shape[0]
    supp_p = supp_p_RHS + supp_p_not_RHS
    supp_RHS = supp_p_RHS + supp_not_p_RHS
    conf = supp / supp_p
    lift = supp_p_RHS / (supp_p*supp_RHS)
    return supp, conf, lift

def association_test(var_list: list, RHS, dataset) -> bool:
    """ An association is significant if the oddsratio is with 95% confidence higher than 1. In this function the association 
    between the set of variables in var_list and the target variable RHS is tested. Returns True when there is a significant
    association and False otherwise
    """
    supp_p_RHS, supp_p_not_RHS, supp_not_p_not_RHS, supp_not_p_RHS = support_counter(var_list, RHS, dataset)
    #Haldane-Anscombe correction for when one of the support is 0 (the correction is just adding 0.5 to all)
    supp_p_RHS, supp_p_not_RHS= supp_p_RHS + 0.5, supp_p_not_RHS + 0.5
    supp_not_p_not_RHS, supp_not_p_RHS = supp_not_p_not_RHS + 0.5 , supp_not_p_RHS + 0.5
    
    #95% confidence -> RHS' = 1.96, odr = oddsratio(p -> RHS) and lb = lower bound
    odr =(supp_p_RHS * supp_not_p_not_RHS)/(supp_p_not_RHS * supp_not_p_RHS)
    
    lb_or= math.exp(np.log(odr)- (1.96*(math.sqrt((1/supp_p_RHS) + (1/supp_p_not_RHS) + (1/supp_not_p_RHS) + (1/supp_not_p_not_RHS)))))
    if lb_or > 1:
        return True
    else:
        return False

def find_exclusive_variables(LHS:list, X:list, dataset, min_l_supp = 0) -> list:
    """
    Requires input of the lefthand side of the association rule and the list of variables which can be exclusive from the 
    lefthand side (X). For each element in X it is tested whether the element is exclusive from the LHS list. The function 
    returns the set of variables that are exclusive in list E. 
    """
    
    E = []
    P = LHS
    if P==[]:
        return []
    p = 1
    for v in LHS:
        p = p & (dataset[v] ==1)          

    for variable in X:
        Q = variable
        if Q not in P:
            q = (dataset[Q] == 1)
            mask_p_q = p & q
            mask_not_p_q = (~p) & q
            supp_p_q = dataset[mask_p_q][Q].count()
            supp_not_p_q = dataset[mask_not_p_q][Q].count()
            if supp_p_q <= min_l_supp:
                E.append(Q)
            elif supp_p_q <= min_l_supp:
                E.append(Q)  
    return E


def create_fair_dataset_original(dataset, LHS_set:list, RHS, C:list):
  
    all_relevant_vars = C + [RHS] + LHS_set
    
    n12, n21 = 0, 0
    
    P = LHS_set

    df_C = dataset[all_relevant_vars]

    C_mask=1
    for c in C:
        C_mask = C_mask & (df_C[c]==1)
    df_C=df_C[C_mask]
    p =1
    for i in P:
        p = p & (df_C[i] == 1)
    df_p = df_C[all_relevant_vars][p]
    df_not_p = df_C[all_relevant_vars][~p]
    if df_p.count()[0]+df_not_p.count()[0]!=df_C.count()[0]:
        print("missing rows counted")
        exit()
    count = min (df_p.count()[0],df_not_p.count()[0])
    if count==0:
        print('empty data set')
    df_p_sample = df_p.sample(count)
    df_not_p_sample = df_not_p.sample(count)
    #                  | not p, RHS   + not p, not RHS  
    #       -----------------------------------------
    #       p, RHS       | n11         n12
    #       p, not RHS   | n21         n22
    #       
    pRHS_count= df_p_sample[RHS].sum()
    pnotRHS_count = count - df_p_sample[RHS].sum()
    notpRHS_count= df_not_p_sample[RHS].sum()
    notpnotRHS_count= count - df_not_p_sample[RHS].sum()
    if (pRHS_count <= notpnotRHS_count): 
        n12+=pRHS_count               # there are at most pRHS_count possible matched pairs containing RHS in exposure group (p) and not RHS in non-exposure group (not p)
        n21+=notpRHS_count            # there are at most notpRHS_count possible matched pairs containing not RHS in exposure group (p) and RHS in non-exposure group (not p)
    else:
        n12+=notpnotRHS_count         # there are at most notpnotRHS_count possible matched pairs containing RHS in exposure group (p) and not RHS in non-exposure group (not p)
        n21+=pnotRHS_count            # there are at most pnotRHS_count possible matched pairs containing not RHS in exposure group (p) and RHS in non-exposure group (not p)

    return n12, n21


def create_fair_dataset(dataset, LHS_set:list, RHS, C:list):
    """
    Matches tuples and returns n12, n21
    """
    all_relevant_vars = C + [RHS] + LHS_set
    fair_dataset=pd.DataFrame(columns=all_relevant_vars)
    
    n12, n21 = 0, 0
    
    P = LHS_set

    df_C = dataset[all_relevant_vars].groupby(C)
    for name, df_c in df_C:
        p =1
        for i in P:
            p = p & (df_c[i] == 1)
        df_p = df_c[all_relevant_vars][p]
        df_not_p = df_c[all_relevant_vars][~p]
        if df_p.count()[0]+df_not_p.count()[0]!=df_c.count()[0]:
            print("missing rows counted")
            exit()
        count = min (df_p.count()[0],df_not_p.count()[0])
        if count==0:
            continue
        df_p_sample = df_p.sample(count)
        fair_dataset= fair_dataset.append(df_p_sample, ignore_index=True)
        df_not_p_sample = df_not_p.sample(count)
        fair_dataset= fair_dataset.append(df_not_p_sample, ignore_index=True)

        #                  | not p, RHS   + not p, not RHS  
        #       -----------------------------------------
        #       p, RHS       | n11         n12
        #       p, not RHS   | n21         n22
        #       
        pRHS_count= df_p_sample[RHS].sum()
        pnotRHS_count = count - df_p_sample[RHS].sum()
        notpRHS_count= df_not_p_sample[RHS].sum()
        notpnotRHS_count= count - df_not_p_sample[RHS].sum()
        if (pRHS_count <= notpnotRHS_count): 
            n12+=pRHS_count               # there are at most pRHS_count possible matched pairs containing RHS in exposure group (p) and not RHS in non-exposure group (not p)
            n21+=notpRHS_count            # there are at most notpRHS_count possible matched pairs containing not RHS in exposure group (p) and RHS in non-exposure group (not p)
        else:
            n12+=notpnotRHS_count         # there are at most notpnotRHS_count possible matched pairs containing RHS in exposure group (p) and not RHS in non-exposure group (not p)
            n21+=pnotRHS_count            # there are at most pnotRHS_count possible matched pairs containing not RHS in exposure group (p) and RHS in non-exposure group (not p)

    return n12, n21, fair_dataset



# based on the paper From Observational Studies to Causal Rule Mining, by Li et al., ACM TIST 7(2), 2015
def CRCS(data, variables, RHS, min_l_supp, max_k):
    #1
    Rc = []

    #2
    main_root = Node('Null',{},[],0)

    for v in variables:
        n=Node(v,{v},[main_root],1)
        main_root.children.add(n)
        n.lsupp =local_support({v},RHS,data)


    #4
    prune(main_root,min_l_supp)

    X = [ c.itemName for c in main_root.children]

    I = set()
    for variable in X:
        #If the function does not return True for RHS(Z):
        if not association_test([variable], RHS, data):
            I.add(variable)
    #7 
    k = 1
    #8
    while k <= max_k:
        found= False
    #9 LHS_AR stores the sets of variables of the LHS of the k-th level
        LHS_AR= generateAssocationRules(main_root,RHS,k,data)

    #10
        for LHS in LHS_AR:

    #11
            E = find_exclusive_variables(LHS, X, data)
    #12     
            C = [variable for variable in X if variable not in I 
                                            if variable not in E
                                            if variable not in LHS]
    #13
            n12, n21, df_fair = create_fair_dataset(data, LHS, RHS, C)
    #14 calculate Odds Ratio, page 10 if zero > count = 1 to evade infinite odds ratios
            if n21 == 0:
                n21 = 1
            if n12 == 0:
                n12 = 1
                
            O_ratio = n12/n21
            # if lowerbound odds ratio (lb_or) > 1 with confidence 95% (z = 1.96)
            lb_or = math.exp(np.log(O_ratio)- (1.96*(math.sqrt((1/(n12)) + (1/(n21))))))
            if lb_or > 1:
    #15         
                found= True
                Rc.append(LHS)
    #16
                removeNode(main_root,set(LHS))
                n=contain(main_root,main_root,set(LHS))
                if n!=None:
                    print("Error: tree still contains removed node!")
                    n.display(1)
                    exit()
                rules_or[tuple(LHS)]=O_ratio
                #df_fair.to_csv("fair_n_"+str(LHS)+".csv", sep=";", index=False)  # if a witness is needed


        if not found:
            print ("No CR rules found at level ",k)

        # No nodes should be added after the max tree depth has been reached
        if k == max_k:
            break
        
        print('\n'+'adding level ', k, 'nodes to the tree') 

    #20
        generateNextLevelTree(main_root,main_root,k,data,RHS) 
    #21 & 22
        prune(main_root,min_l_supp)
    #23    
        pruneSameSupport(main_root)
        k += 1
    return Rc

def mine_causal_rules():
    RHS='Z' # for mining positive causal rules
    max_k = 5
    pRc=CRCS(data,new_selected_variables,RHS,min_l_supp,max_k) 
    RHS='notZ' # for mining negative causal rules
    nRc=CRCS(data,new_selected_variables,RHS,min_l_supp,max_k)
    return pRc,nRc

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


def printrules(pRc,nRc,cvl,data):
    df_rules = pd.DataFrame(columns = ['Rule', 'Controllable', 'Odds ratio',  'Support'])
    for r in pRc:
        s,c,l=computeMetrics(r,'Z',data)
        rule= str(r)+" --> Z"
        b=iscontrollable(r,cvl)
        df_rules = df_rules.append({'Rule':rule,'Controllable':b,'Odds ratio': rules_or[tuple(r)],'Support':s },ignore_index = True)
    for r in nRc:
        s,c,l=computeMetrics(r,'notZ',data)
        rule= str(r)+" --> !Z"
        b=iscontrollable(r,cvl)
        df_rules = df_rules.append({'Rule':rule,'Controllable':b,'Odds ratio': rules_or[tuple(r)],'Support':s },ignore_index = True)    
    df_rules.to_csv("rules_"+name+".csv",sep=";",index=False)



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

min_l_supp = 0.01 
rules_or = {} # odds ratio



prc_occ=dict()
odds=dict()
nrc_occ=dict()
for i in range(1,11):
    print("iteration "+str(i))
    prc, nrc= mine_causal_rules()
    for r in prc:
        if  tuple(r) in prc_occ:
            prc_occ[tuple(r)]=prc_occ[tuple(r)]+1
            odds[tuple(r)]=odds[tuple(r)]*((prc_occ[tuple(r)]-1)/prc_occ[tuple(r)]) + (1/prc_occ[tuple(r)])*rules_or[tuple(r)]
        else:
            prc_occ[tuple(r)]=1
            odds[tuple(r)]=rules_or[tuple(r)]

    for r in nrc:
        if  tuple(r) in nrc_occ:
            nrc_occ[tuple(r)]=nrc_occ[tuple(r)]+1
            odds[tuple(r)]=odds[tuple(r)]*((nrc_occ[tuple(r)]-1)/nrc_occ[tuple(r)]) + (1/nrc_occ[tuple(r)])*rules_or[tuple(r)]
        else:
            nrc_occ[tuple(r)]=1
            odds[tuple(r)]=rules_or[tuple(r)]


pRc_to_evalute=list()
nRc_to_evalute=list()
for k,i in prc_occ.items():
    if i >7:
        pRc_to_evalute.append(list(k))
for k,i in nrc_occ.items():
    if i >7:
        nRc_to_evalute.append(list(k))



printrules(pRc_to_evalute,nRc_to_evalute,new_controllable_variables,data)
