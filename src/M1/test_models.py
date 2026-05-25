""" 
Run tests for all models and store the results.
Some of the code does not work as it was only used during the development phase to find the best models and parameters, but it is kept for reference.
"""

from utils import *
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_predict, ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from convMixer import convmixer_loop

class Tests:
    """ Class to handle all the tests """
    def __init__(self, X, y, kf):
        self.X = X
        self.y = y
        self.y_pred = None
        self.kf = kf
        self.errors_graph = []
        self.errors_tabular = []
        self.names_graph = []
        self.names_tabular = []
        self.all_models_names = []
        self.results = []
        self.test_models()

    def get_errors(self):
        return self.errors_graph, self.names_graph, self.errors_tabular, self.names_tabular

    def test_models(self):
        """ Run all tests """
        self.test_linear_reg()
        print(self.results)
        self.test_ridge_reg()
        self.test_lasso_reg()
        self.test_knn()
        self.test_decision_tree()
        self.test_svm()
        self.test_gradient_boosting()
        self.test_random_forest()
        self.test_convmixer()

    def add_for_graph_latex(self, best_nmse, names, errors, names_errors):
        """ Add the best NMSE values for each metric to the errors and names lists """
        self.errors_graph.append(best_nmse['NMSE'])
        self.names_graph.append(names['NMSE'])
        for metric in ['NMSE', 'NMSE_x', 'NMSE_y', 'NMSE_z']:
            error, name = get_error_from_name(names[metric], errors, self.all_models_names)
            self.errors_tabular.append(error)
            self.names_tabular.append(name)
        return errors, names_errors

    def test_linear_reg(self):# Initialize parameters
        fit_intercepts = [True, False]  # Whether to fit an intercept term in the regression model.

        # Call Linear Regression loop (without alpha)
        print("---- Linear ----")
        values = ["Linear Regression"]
        values.extend(self.linear_regression_loop(fit_intercepts))
        self.results.append(values)
        # best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        # self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_ridge_reg(self):
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]  # Regularization strength for Ridge and Lasso. Higher alpha means more regularization.
        fit_intercepts = [True, False]  # Whether to fit an intercept term in the regression model.
        # Call Ridge loop (with alpha)
        print("---- Ridge ----")
        errors, model_names = self.ridge_loop(alphas, fit_intercepts)
        best_nmse, names = find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_lasso_reg(self):
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]  # Regularization strength for Ridge and Lasso. Higher alpha means more regularization.
        fit_intercepts = [True, False]
        print("---- Lasso ----")
        errors, model_names = self.lasso_loop(alphas, fit_intercepts)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_knn(self):
        # Run KNN tuning
        print("---- KNN ----")

        # KNN Regressor parameters grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 10],  # Number of neighbors to consider for predictions
            'weights': ['uniform', 'distance'],  # 'uniform' = equal weight to all neighbors, 'distance' = closer neighbors have more influence
            'algorithm': ['ball_tree', 'kd_tree'],  # Algorithm used for nearest neighbor search
            'leaf_size': [10, 20, 30],  # Leaf size for BallTree/KDTree (affects speed vs. memory tradeoff)
            'p': [1, 2],  # Distance metric: 1 = Manhattan, 2 = Euclidean
        }

        errors, model_names = self.knn_loop(param_grid)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_decision_tree(self):
        # Run Decision Tree tuning
        print("---- Decision Tree ----")

        # Define hyperparameter grid
        param_grid = {
            'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree. 'None' means the nodes are expanded until all leaves are pure.
            'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node. 
            'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node.
            'criterion': ['squared_error', 'absolute_error'],  # The function to measure the quality of a split: 'mse' (mean squared error) or 'mae' (mean absolute error).
            'max_features': [None, 'sqrt', 'log2'],  # The number of features to consider when looking for the best split: 'sqrt' and 'log2' limit the number of features to the square root or logarithm of the total number of features.
        }

        errors, model_names = self.decision_tree_loop(param_grid)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_svm(self):
        # Run SVM tuning
        print("---- SVM ----")

        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Regularization parameter: Higher C values lead to less regularization
            'kernel': ['linear', 'poly', 'rbf'],  # Type of kernel function used to compute the decision boundary
            'degree': [3, 5, 10],  # Degree of the polynomial kernel function (used only if kernel is 'poly')
            'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'linear'. 'scale' is 1 / (n_features * X.var()) and 'auto' is 1 / n_features
            'epsilon': [0.1, 0.2, 0.5],  # Epsilon parameter for the epsilon-insensitive loss function
            'shrinking': [True, False],  # Whether to use the shrinking heuristic to speed up convergence
        }
        errors, model_names = self.svm_loop(param_grid)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_gradient_boosting(self):
        print("---- Gradient Boosting ----")

        # Hyperparameters to tune
        n_estimators_list = [10, 50, 100]  # Number of boosting stages to run. More estimators can improve performance but may lead to overfitting.
        learning_rates = [0.01, 0.1, 0.2]  # The step size at each iteration while moving towards a minimum of the loss function. Lower values can improve generalization but may require more estimators.
        max_depths = [3, 5, 7]  # Maximum depth of individual trees. A higher value makes the model more complex and increases the risk of overfitting.

        errors, model_names = self.gradient_boosting_loop(n_estimators_list, learning_rates, max_depths)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_random_forest(self):
        print("---- Random Forest ----")
        # Hyperparameters to tune
        n_estimators_list = [10,50, 100]  # The number of trees in the forest. More trees typically improve performance, but at the cost of increased computation time.
        max_depths = [None, 10, 20]  # The maximum depth of each tree. 'None' means nodes are expanded until all leaves are pure.
        min_samples_splits = [2, 5, 10]  # The minimum number of samples required to split an internal node.
        min_samples_leaves = [1, 2, 5]  # The minimum number of samples required to be at a leaf node.


        errors, model_names = self.random_forest_loop(n_estimators_list, max_depths, min_samples_splits, min_samples_leaves)
        best_nmse, names =find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print("Done")

    def test_convmixer(self):
        print("---- ConvMixer ----")
        # Define ConvMixer hyperparameter grid
        dims = [16, 32]
        depths = [4, 6]
        kernel_sizes = [3]
        patch_sizes = [4, 7]
        learning_rates = [1e-4,1e-3, 5e-3]
        n_epochs_list = [50, 100, 150]

        # Run the loop
        errors, model_names = convmixer_loop(self.X, self.y, self.kf, dims, depths, kernel_sizes, patch_sizes, learning_rates, n_epochs_list)
        self.all_models_names.extend(model_names)  # Append ConvMixer model names to all_models_names

        # Append best results
        best_nmse, names = find_best_model_for_each_metric(errors, model_names)
        self.add_for_graph_latex(best_nmse, names, errors, model_names)
        print('Done')



    def linear_regression_loop(self, fit_intercepts)-> list:
        best = [999]
        for fit_intercept in fit_intercepts:
            model = LinearRegression(fit_intercept=fit_intercept)
            y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
            nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)
            if nmse < best[0]:
                best = [nmse, nmse_x, nmse_y, nmse_z, f"Linear Regression (fit intercept={fit_intercept})"]
        # errors, model_names = [], []
        # # Linear Regression (no alpha parameter)
        # for fit_intercept in fit_intercepts:
        #     model = LinearRegression(fit_intercept=fit_intercept)
        #     y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
            
        #     # Compute NMSE for overall and per coordinate
        #     nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)
            
        #     # Store results
        #     errors.append([nmse, nmse_x, nmse_y, nmse_z])
        #     model_names.append(f"Linear Regression (fit intercept={fit_intercept})")
        #     self.all_models_names.append(model_names[-1])
        
        return best #errors, model_names

    def ridge_loop(self, alphas, fit_intercepts):
        errors, model_names = [], []
        for alpha in alphas:
            for fit_intercept in fit_intercepts:
                model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
                y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
                
                # Compute NMSE for overall and per coordinate
                nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)
                
                # Store results
                errors.append([nmse, nmse_x, nmse_y, nmse_z])
                model_names.append(f"Ridge (alpha={alpha}, fit intercept={fit_intercept})")
                self.all_models_names.append(model_names[-1])
        
        return errors, model_names

    def lasso_loop(self, alphas, fit_intercepts):
        errors, model_names = [], []
        for alpha in alphas:
            for fit_intercept in fit_intercepts:
                model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
                y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
                
                # Compute NMSE for overall and per coordinate
                nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)
                
                # Store results
                errors.append([nmse, nmse_x, nmse_y, nmse_z])
                model_names.append(f"Lasso (alpha={alpha}, fit intercept={fit_intercept})")
                self.all_models_names.append(model_names[-1])
        
        return errors, model_names


    def gradient_boosting_loop(self, n_estimators_list, learning_rates, max_depths):
        errors, model_names = [], []
        for n_estimators in n_estimators_list:
            print(n_estimators)
            for learning_rate in learning_rates:
                print(learning_rate)
                for max_depth in max_depths:
                    # Define Gradient Boosting model
                    base_model = GradientBoostingRegressor(
                        n_estimators=n_estimators, 
                        learning_rate=learning_rate, 
                        max_depth=max_depth, 
                        random_state=42
                    )
                    model = MultiOutputRegressor(base_model)

                    # Cross-validation prediction
                    y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)

                    # Compute NMSE for overall and per coordinate
                    nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)

                    # Store results
                    errors.append([nmse, nmse_x, nmse_y, nmse_z])
                    model_names.append(f"Gradient Boosting (n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth})")
                    self.all_models_names.append(model_names[-1])
        return errors, model_names



    def random_forest_loop(self, n_estimators_list, max_depths, min_samples_splits, min_samples_leaves):
        errors, model_names = [], []
        
        for n_estimators in n_estimators_list:
            print(n_estimators)
            for max_depth in max_depths:
                print(max_depth)
                for min_samples_split in min_samples_splits:
                    for min_samples_leaf in min_samples_leaves:
                        # Define Random Forest model
                        base_model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42
                        )
                        model = MultiOutputRegressor(base_model)

                        # Cross-validation prediction
                        y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)

                        # Compute NMSE for overall and per coordinate
                        nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)

                        # Store results
                        errors.append([nmse, nmse_x, nmse_y, nmse_z])
                        model_names.append(f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf})")
                        self.all_models_names.append(model_names[-1])
        return errors, model_names

    def decision_tree_loop(self, param_grid):
        errors, model_names = [], []
        for params in ParameterGrid(param_grid):
            # Define Decision Tree model with current parameters
            model = DecisionTreeRegressor(**params)
            
            # Perform cross-validation
            y_pred = cross_val_predict(model, self.X,self.y, cv=self.kf)
            
            # Compute NMSE for overall and per coordinate
            nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)
            
            # Store results
            errors.append([nmse, nmse_x, nmse_y, nmse_z])
            model_names.append(f"Decision Tree {params}")
            self.all_models_names.append(model_names[-1])
        return errors, model_names

    def svm_loop(self, param_grid):
        errors, model_names = [], []
        for params in ParameterGrid(param_grid):
            # Define SVM model with current parameters and MultiOutputRegressor
            model = MultiOutputRegressor(SVR(**params))

            # Perform cross-validation separately for each output dimension
            y_pred = np.zeros_like(self.y)
            for i in range(self.y.shape[1]):  # Iterate over each output dimension
                y_pred[:, i] = cross_val_predict(SVR(**params), self.X, self.y[:, i], cv=self.kf)

            # Compute NMSE for overall and per coordinate
            nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)

            # Store results
            errors.append([nmse, nmse_x, nmse_y, nmse_z])
            model_names.append(f"SVM {params}")
            self.all_models_names.append(model_names[-1])
        return errors, model_names

    def knn_loop(self, param_grid):
        errors, model_names = [], []
        for params in ParameterGrid(param_grid):
            # Define KNN model with current parameters
            model = KNeighborsRegressor(**params)

            # Perform cross-validation
            y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)

            # Compute NMSE for overall and per coordinate
            nmse,nmse_x,nmse_y,nmse_z = NMSE_by_coordinate(self.y, y_pred)

            # Store results
            errors.append([nmse, nmse_x, nmse_y, nmse_z])
            model_names.append(f"KNN {params}")
            self.all_models_names.append(model_names[-1])
        return errors, model_names
        