import pandas as pd
import numpy as np
from utils import *
from distnn import *
from pyomo import environ
from pyomo.environ import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()


class distcl(object):
    '''
    X, y for fitting a Distributional Neural Network are expected.
    n_preds: number of predictions needed in the optimization problem.
    '''
    
    def __init__(self, X, y, n_preds, val_ind = None, test_ind = None):
        
        try:
            val_ind
        except NameError:
            ind = False
        else:
            ind = True
        
        X = X.reindex(sorted(X.columns), axis=1)
        
        if ind == True:
            self.random_split = False
            
            self.X_val = X[X.index.isin(val_ind)]
            self.X_test = X[X.index.isin(test_ind)]
            self.X_train = X[(~X.index.isin(val_ind)) & (~X.index.isin(test_ind))]
            
            self.y_val = y[y.index.isin(val_ind)]
            self.y_test = y[y.index.isin(test_ind)]
            self.y_train = y[(~y.index.isin(val_ind)) & (~y.index.isin(test_ind))]
        else:
            self.random_split = True
            
            msk = np.random.rand(len(X)) < 0.7
            self.X_train = X[msk]
            self.y_train = y[msk]
            self.X_test = X[~msk]
            self.y_test = y[~msk]
            
            msk_val = np.random.rand(len(self.X_test)) < 0.5
            self.X_val = self.X_test[msk_val]
            self.y_val = self.y_test[msk_val]
            self.X_test = self.X_test[~msk_val]
            self.y_test = self.y_test[~msk_val]
            
            #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            #self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=0)
        
        
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns = scaler.feature_names_in_)
        
        self.X_mean = scaler.mean_
        self.X_std = scaler.scale_
        
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns = scaler.feature_names_in_)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns = scaler.feature_names_in_)
        
        self.y_train = pd.DataFrame(scaler.fit_transform(self.y_train.values.reshape(-1, 1)), columns = [y.name])
        self.y_mean = scaler.mean_
        self.y_std = scaler.scale_
        
        self.y_val = pd.DataFrame(scaler.transform(self.y_val.values.reshape(-1, 1)), columns = [y.name])
        self.y_test = pd.DataFrame(scaler.transform(self.y_test.values.reshape(-1, 1)), columns = [y.name])
        
        self.n_preds = n_preds
        self.alpha = None
        
    
    def train(self, n_hidden = 2, n_nodes = 50, drop = 0.05, iters = 4000, learning_rate = 1e-3):
        nn_tool = qdnn(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, n_hidden = n_hidden, n_nodes = n_nodes, drop = drop, iters = iters, learning_rate = learning_rate)
        model, preds_test, vars_test, y_test = nn_tool.train()
        
        return model, preds_test, vars_test, y_test
    
    
    def constraint_build(self, fitted_model):
        
        weight_names = []
        weight_values = []
        bias_names = []
        bias_values = []
        for name, param in fitted_model.named_parameters():
            if 'weight' in name:
                weight_names.append(name)
                weight_values.append(param.cpu().detach().numpy())
            else:
                bias_names.append(name)
                bias_values.append(param.cpu().detach().numpy())
        
        constraints = constraint_extrapolation_MLP(weight_values,bias_values,weight_names)
        
        return constraints
    
    
    def const_embed(self, opt_model, constaints, outcome, n_scenarios = 1, deterministic = False):
        '''
        This function embdeds the fitted prediction model within the optimization problem.
        Expecting a defined optimization model "opt_model", constraint dataframe "constraints",
        a name for an "outcome", and the number of scenarios to generate.
        '''
        
        # Predefining variable y
        #opt_model.y = Var(Any, dense=False, domain=Reals)
        
        M_l=-1e3
        M_u=1e3
        opt_model.v = Var(Any, dense=False, domain=NonNegativeReals)
        opt_model.v_ind = Var(Any, dense=False, domain=Binary)
        
        for n in range(self.n_preds):
            max_layer = max(constaints['layer'])
            nodes_input = range(len(self.X_train.columns))
            v_input = [(opt_model.x[name,n]-self.X_mean[i])/self.X_std[i] for i,name in enumerate(self.X_train.columns)]
            
            for l in range(max_layer):
                df_layer = constaints.query('layer == %d' % l)
                max_nodes = [k for k in df_layer.columns if 'node_' in k]
                # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
                coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                intercepts_layer = np.array(df_layer['intercept'])
                nodes = df_layer['node']
                
                if l == max_layer-1:
                    node = nodes.iloc[0] 
                    opt_model.add_component('Mean_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'mean'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                    l = l+1
                    df_layer = constaints.query('layer == %d' % l)
                    max_nodes = [k for k in df_layer.columns if 'node_' in k]
                    coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                    intercepts_layer = np.array(df_layer['intercept'])
                    opt_model.add_component('Var_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'sd'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                else:
                    # Save v_pos for input to next layer
                    v_pos_list = []
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list.append(opt_model.v[(outcome, l, node, n)])
                        opt_model.add_component('constraint_1_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('constraint_2_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= M_u * (opt_model.v_ind[(outcome, l, node, n)])))
                        opt_model.add_component('constraint_3_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - opt_model.v_ind[(outcome, l, node, n)])))
                    ## Prepare nodes_input for next layer
                    nodes_input = nodes
                    v_input = v_pos_list
        
        
        if deterministic == False:
            np.random.seed(0)
            z_vals = np.random.normal(size=n_scenarios)
            
            for n in range(self.n_preds):
                for w in range(1, n_scenarios + 1):
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == (z_vals[w-1] * opt_model.y[outcome, n,'sd'] + opt_model.y[outcome, n,'mean'])*self.y_std + self.y_mean))
        else:
            for n in range(self.n_preds):
                for w in range(1, n_scenarios + 1):
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == opt_model.y[outcome, n,'mean']*self.y_std + self.y_mean))