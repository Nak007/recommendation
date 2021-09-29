'''
Available methods are the followings:
[1] RSGD
[2] rating_train_test
[3] learning_rate
[4] RecommenderScores
[5] ItemReccommender

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 01-10-2021

'''
import pandas as pd, numpy as np, numbers
from itertools import product
import collections
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (dcg_score, ndcg_score)
import time, datetime
import ipywidgets as widgets
from IPython.display import display
import numbers

__all__ = ["RSGD", "learning_rate",
           "rating_train_test",
           "RecommenderScores", 
           "ItemReccommender"]

class RSGD():
    
    '''
    Regularized Stochastic Gradient Descent
    
    `RSGD` implements "Collaborative Filtering" (CF) by using 
    "Matrix Factorization" (MF) that minimizes cost function 
    using gradient descent to uncover latent factor that 
    explain observed ratings. This algorithm is suitable for 
    explicit feebacks.
    
    In addition to the learning of the factor vector P[i] and 
    Q[j] , the system minimizes the regularized squared error 
    on the set of known errors. The system learns the model by 
    fitting the previously observed ratings, so as to predict 
    future unknown ratings [1].

    Parameters
    ----------
    max_iter : int, default=5000
        Maximum number of iterations for a single run. This 
        overrides `lr_kwds`.
        
    alpha : float, default=0.01
        Initial learning rate. This overrides `lr_kwds`.

    beta : float, default=0.01
        Regularization parameter. If `beta` equals to 0, it is 
        equivalent to an ordinary stochastic gradient descent.
        
    biased : bool, default=True
        - True  : r(u,i) = 𝜇 + b(u) + b(i) + Q(i).T.P(u)
        - False : r(u,i) = Q(i).T.P(u)
        where 
            - r(u,i) is a rating of item `i` of user `u`,
            - 𝜇 is global baseline (mean of ratings), 
            - b(u) is bias of user `u`,
            - b(i) is bias of item `i`, 
            - Q(i) is item `i` factors, and
            - P(u) is user `u` factors.

    lr_kwds : keywords, default=None
        Keyword arguments to be passed to "learning_rate".

    min_delta : float, default=1e-4
        Minimum absolute change in loss of `X_val` (if provided,
        otherwise `X`) to be qualified as an improvement, i.e. a 
        change of less than `min_delta`, will count as no 
        improvement and declare convergence.
    
    patience : int, default=5
        Number of iterations with no improvement of `X_val` (if 
        provided, otherwise `X`) after which training will be 
        stopped. 

    proportion : float, default=0.8
        Minimum proportion of variation explained when choosing 
        number of latent factors. Only applicable if `n_factor` 
        is None.
        
    random : {"normal", "uniform"}, default="normal"
        The distribtion for random samples.
        - "normal"  : Normal distribution, N(init_mean, init_std).
        - "uniform" : Uniform distribution over [0, 1).

    init_mean : float, default=0.
        The mean of the normal distribution for factor vectors 
        initialization. This is relevant when `random` is
        "normal".

    init_std : float, default=0.1
        The standard deviation of the normal distribution for 
        factor vectors initialization. This is relevant when 
        `random` is "normal".
    
    baseline : float, default=0.
         The baseline bias for all users. This revelant when 
         `biased` is True.
 
    random_state : int, default=None
        It controls the randomness of creating `user_factors`
        `item_factors`, `user_bias`, and `item_bias` by populating 
        them with random samples.
        
    verbose : bool, default=True
        If True, messages are displayed during optimization.
        
    References
    ----------
    .. [1] Rahul Makhijani, Saleh Samaneh, and Megh Mehta, 
           "Collaborative Filtering Recommender Systems"
    .. [2] https://scikit-learn.org/stable/modules/generated/
           sklearn.decomposition.TruncatedSVD.html
    .. [3] https://towardsdatascience.com/recommendation-
           system-matrix-factorization-d61978660b4b
    .. [4] Yifan Hu, Yehuda Koren, and Chris Volinsky, 
           "Collaborative Filtering for Implicit Feedback Datasets"
    .. [5] https://github.com/fjssharpsword/BayesianRS
    .. [6] https://www.youtube.com/watch?v=YW2b8La2ICo

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[ 1, 1, 1, 0, 0],
                      [ 3, 3, 3, 0, 0],
                      [ 4, 4, 4, 0, 0],
                      [ 5, 5, 5, 0, 0],
                      [ 0, 2, 0, 4, 4],
                      [ 0, 0, 0, 5, 5],
                      [ 0, 1, 0, 2, 2],
                      [ 0, 0, 3, 1, 0]])
    >>> R = np.where(R==0, np.nan, R)
                      
    >>> estimator = RSGD_w_Bias(random_state=0)
    >>> predictions = estimator.fit_predict(R)
    
    >>> np.round(np.fmin(predictions, 5), 2)
    array([[1.01, 0.99, 1.01, 1.44, 1.46],
           [2.99, 2.98, 2.99, 2.9 , 2.93],
           [3.98, 3.98, 3.99, 3.65, 3.68],
           [4.98, 4.98, 4.98, 4.38, 4.42],
           [2.14, 2.  , 1.41, 3.98, 3.98],
           [3.36, 3.22, 2.55, 4.98, 4.99],
           [1.05, 1.01, 0.96, 1.99, 2.  ],
           [2.05, 2.17, 2.98, 1.  , 1.04]])
  
    '''
    def __init__(self, max_iter=5000, alpha=0.01, beta=0.01, biased=True,
                 lr_kwds=None, min_delta=0.0001, patience=5, proportion=0.8, 
                 random="normal", init_mean=0, init_std=0.1, baseline=0, 
                 random_state=None, verbose=True):
        
        self.max_iter = max_iter
        self.alpha = alpha
        kwds = dict(lr0=alpha, n_iters=max_iter)
        lr_kwds = {} if lr_kwds is None else lr_kwds
        self.learning_rates = learning_rate(**{**kwds, **lr_kwds})
        self.beta  = beta
        self.min_delta = min_delta
        self.patience = patience
        self.proportion= proportion
        self.random_state = random_state
        self.verbose = verbose
        self.random = random
        self.sampling = dict(loc=init_mean, scale=init_std)
        self.biased = biased
        self.baseline = 0 if biased else baseline

    def fit_predict(self, X, n_factors=None, X_val=None):
        
        '''
        Compute User-factors, and Item-factors and predict unknown
        ratings for each user-item.
        
        Parameters
        ----------
        X : ndarray, of shape (n_users, n_items)
            User-item observation matrix. Missing value or zero 
            is treated as an unknown (undecidable recommendation) 
            and will be estimated by the algorithm.

        n_factors : int, default=None
            Number of latent factors. If None, the algorithm uses
            "TruncatedSVD" [2] to determine number of factors that
            satisfies minimum variation explained `proportion`.
            
        X_val : ndarray, default=None
            The unseen user-item observation matrix used to provide 
            an unbiased evaluation. `X_val` must be in the same 
            shape as `X`. If None, validation is not implemented.

        Returns
        -------
        prediction : ndarray, of shape (n_users, n_items)
            The prediction is done by adding `baseline`, `user_bias`,
            `item_bias` and an inner product between `user_factors` 
            and `item_factors`. 
        
        References
        ----------
        .. [1] Rahul Makhijani, Saleh Samaneh, and Megh Mehta, 
               "Collaborative Filtering Recommender Systems"
        .. [2] https://scikit-learn.org/stable/modules/generated/
               sklearn.decomposition.TruncatedSVD.html
        .. [3] https://towardsdatascience.com/recommendation-
               system-matrix-factorization-d61978660b4b
        .. [4] Yifan Hu, Yehuda Koren, and Chris Volinsky, 
               "Collaborative Filtering for Implicit Feedback Datasets"
        .. [5] https://github.com/fjssharpsword/BayesianRS
        .. [6] https://www.youtube.com/watch?v=YW2b8La2ICo
        
        Attributes
        ----------
        n_iters : int
            Number of iterations to reach the specified tolerance 
            i.e. `min_delta`, and `patience`.
            
        user_factors : ndarray, of shape (n_users, n_factors)
            Updated User-factors matrix.
            
        item_factors : ndarray, of shape (n_items, n_factors)
            Updated Item-factors matrix.
            
        monitor : collections.namedtuple
            Metrics to be monitored:
        
            [1] loss_train : ndarray, of shape (n_iters,) 
                Objective functions (loss) as per iteration.
                
            [2] rmse_train : ndarray, of shape (n_iters,)
                Root Mean Square Error between `X` and estimates.
            
            [3] loss_val : ndarray, of shape (n_iters,)
                Objective functions (loss) of `X_val` as per iteration.
                This is relevant when `X_val` is provided.
                
            [4] rmse_val : ndarray, of shape (n_iters,)
                Root Mean Square Error between `X_val` and estimates.
                This is relevant when `X_val` is provided.
        
        bias : collections.namedtuple
            
            [1] baseline : float
                Average rating of all users.
    
            [2] user_bias : ndarray, of shape (n_users,)
                Bias of user u.
         
            [3] item_bias : ndarray, of shape (n_items,)
                Bias of item i.
        
        '''
        # Initialize Widget for verbosity
        w1, w2 = InitializeWidget(self.verbose)
        start  = time.time()
        
        # Truncated SVD
        if n_factors is None:
            kwds = {"n_components" : X.shape[1]-1, 
                    "random_state" : self.random_state}
            svd = TruncatedSVD(**kwds).fit(np.nan_to_num(X, nan=0.))
            varprop = np.cumsum(svd.explained_variance_ratio_)
            n_factors = sum(varprop<self.proportion) + 1
            
        # Randomize P, and Q, where R = P.Q 
        # P : |U| * K (User feature matrix)
        # Q : |I| * K (Item feature matrix)
        # K : Number of latent features
        np.random.seed(self.random_state)
        n_users, n_items = X.shape
        if self.random=="normal":
            P = np.random.normal(size=(n_users, n_factors), **self.sampling)
            Q = np.random.normal(size=(n_items, n_factors), **self.sampling).T
        elif self.random=="uniform":
            P = np.random.rand(n_users, n_factors)
            Q = np.random.rand(n_items, n_factors).T
        else: raise ValueError(f"The algorithm supports only 'normal', and " 
                               f"'uniform'. Got {self.random} instead.")
            
        # Initialize the biases (dtype=float64)
        if self.biased & (self.random=="normal"):
            user_bias = np.random.normal(size=n_users, **self.sampling)
            item_bias = np.random.normal(size=n_items, **self.sampling)
        elif self.biased & (self.random=="uniform"):
            user_bias = np.random.rand(n_users)
            item_bias = np.random.rand(n_items)
        else: 
            user_bias = np.zeros(n_users) 
            item_bias = np.zeros(n_items) 
        
        # Validate `X_val`
        rmse = [[], ValidateTest(X, X_val)]
        loss = [[], ValidateTest(X, X_val)]
        
        # Iteration pairs (i,j). X(i,j)=np.nan (unknown) is ignored.
        iters = np.array(list(product(range(n_users), range(n_items))))
        iters = [iters[~(np.isnan(X).ravel())], 
                 (None if X_val is None else iters[~(np.isnan(X_val).ravel())])]
        
        # Initialize parameters
        n_iter, data = 1, [X, X_val]
        eps  = np.finfo(float).eps
        nval = 0 if X_val is None else 1
        best_P, best_Q, current_best, patience = None, None, np.inf, 0
        
        while True:
                
            # Update User-factors, and Item-factors matrix
            e = X - (self.baseline + P.dot(Q) +
                     user_bias.reshape(-1, 1) + 
                     item_bias.reshape( 1,-1))

            # Update learning rate
            alpha = self.learning_rates[n_iter-1]
         
            for (i,j) in iters[0]:
                
                # Update biases (user & item) as follows
                # bu[i] = bu[i] + 2 * α(e[i,j] − β * bu[i])
                # bi[j] = bi[j] + 2 * α(e[i,j] − β * bi[j])
                # Note: 2 is ignored since α is very small.
                user_bias[i] += (alpha * (e[i,j] - self.beta * user_bias[i]) 
                                 if self.biased else 0)
                item_bias[j] += (alpha * (e[i,j] - self.beta * item_bias[j]) 
                                 if self.biased else 0)
                
                # Calculate gradient and update P, and Q as follows:
                # P[i,k] = P[i,k] + 2 * α(e[i,j] * Q[k,j] − β * P[i,k]) 
                # Q[k,j] = Q[k,j] + 2 * α(e[i,j] * P[i,k] − β * Q[k,j])
                # Note: 2 is ignored since α is very small.
                P[i,:] += alpha * (e[i,j] * Q[:,j] - self.beta * P[i,:])
                Q[:,j] += alpha * (e[i,j] * P[i,:] - self.beta * Q[:,j])
            
            # Check wheter explosive solution occurs or not
            for a in [user_bias, item_bias, P, Q]:
                have_nan = np.isnan(a).sum()>0
                have_inf = np.isin(a, [-np.inf, np.inf]).sum()>0
                if have_nan | have_inf:
                    raise ValueError("Explosive solution: some numeric values " 
                                     "overflow the range of (-∞, ∞). Lowering the "
                                     "inital learning rate (`alpha`) might solve "
                                     "this problem. Got {:,.2g}.".format(self.alpha))
            
            # Estimates = μ + user_bias + item_bias - (P.T).Q
            estimate = (self.baseline + P.dot(Q) + 
                        user_bias.reshape(-1, 1) + 
                        item_bias.reshape( 1,-1))
            
            # Minimize cost function, min ∑ pow(e,2)
            # e[i,j]^2 = (R[i,j] − μ - bu[i] - bi[j] - P[i].Q[j])^2
            for n in range(2):
                
                if data[n] is not None:
                    
                    # Root mean square errors.
                    sse = np.nansum((data[n] - estimate)**2)
                    mse = np.nanmean((data[n]- estimate)**2)

                    # Regulariziation
                    # l2 = β * ∑(||P[i]||^2 + ||Q[j]||^2 + bu[i]^2 + bi[j]62)
                    l2 = self.beta * sum([sum(pow(P[i,:], 2)) + 
                                          sum(pow(Q[:,j], 2)) +
                                          pow(user_bias[i],2) + 
                                          pow(item_bias[j],2) 
                                          for (i,j) in iters[n]])
                    
                    rmse[n] += [np.sqrt(mse)]
                    loss[n] += [sse + l2]
            
            # Check whether the result has converged.
            if n_iter > 1: 
                delta = abs(np.diff(loss[nval][-2:])[0])
                converge = (delta < self.min_delta)
            else: converge = False
        
            # Keep the best P and Q
            if loss[nval][-1] < current_best:
                best_P = P.copy()
                best_Q = Q.copy()
                best_user_bias = user_bias.copy()
                best_item_bias = item_bias.copy()
                current_best = loss[nval][-1]
                patience = 0
            else: patience += 1
            
            # Check whether the result has continued improving.
            improve = patience < self.patience
         
            # Algorithm stops when either result stops improving 
            # (lower loss) or convergence happens.
            if (improve==False) | (converge==True) | (n_iter==self.max_iter): 
                if self.verbose:
                    elapse = int(time.time() - start) 
                    elapse = str(datetime.timedelta(seconds=elapse))
                    w1.value = f"Complete, Elapsed Time : [{elapse}] >>> "
                    UpdateWidget(w2, n_iter, rmse[nval][-1], loss[nval][-1]) 
                    
                    # Stopping criteria
                    criteria = ["Converge  : {:.2g} (Max={:.2g})"
                                .format(delta, self.min_delta),
                                "Patience  : {:,d} (Max={:,d})"
                                .format(patience, self.patience), 
                                "Iteration : {:,d} (Max={:,d})"
                                .format(n_iter, self.max_iter)]
                    max_len = max(criteria, key=len).__len__()
                    
                    print("Stopping Criteria")
                    print("=" * max_len)
                    print(("\n" + "-"* max_len + "\n").join(criteria))
                    print("=" * max_len)
      
                break
            else: n_iter += 1

            if self.verbose:
                w1.value = 'Calculating . . . '
                UpdateWidget(w2, n_iter, rmse[nval][-1], loss[nval][-1])
        
        # Attributes
        self.n_factors = n_factors
        self.n_iters = n_iter
        self.user_factors = best_P
        self.item_factors = best_Q.T
        self.iters = iters
        
        keys = ["loss_train", "rmse_train", "loss_val", "rmse_val"]
        monitor = collections.namedtuple('Monitor', keys)
        self.monitor = monitor(loss_train = np.array(loss[0]), 
                               rmse_train = np.array(rmse[0]), 
                               loss_val   = np.array(loss[1]), 
                               rmse_val   = np.array(rmse[1]))
        
        keys = ["baseline", "user_bias", "item_bias"]
        BIAS = collections.namedtuple('BIAS', keys)
        self.bias = BIAS(baseline  = self.baseline, 
                         user_bias = best_user_bias, 
                         item_bias = best_item_bias)
        
        return  (self.baseline + best_P.dot(best_Q) + 
                 best_user_bias.reshape(-1, 1) + 
                 best_item_bias.reshape( 1,-1))
    
    def plotting(self, monitor="rmse", ax=None, colors=None, 
                 plot_kwds=None, tight_layout=True, val_format=None):
        
        '''
        Plot monitored metrics i.e. loss or rmse.
    
        Parameters
        ----------
        estimator : estimator object
            Fitted `Ordinary_RSGD` or `RSGD_w_Bias` estimator.

        monitor : {"loss", "rmse"}, default="rmse"
            Metric to be monitored.

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created 
            with default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be equal to 2 i.e. Train and
            Test. If None, it uses default colors from Matplotlib.
            This overrides `plot_kwds`.

        plot_kwds : keywords, default=None
            Keyword arguments to be passed to `ax.plot`.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().

        val_format : string formatter, default=None
            String formatters (function) for minimum value of metric 
            of interest to be displayed on labels e.g. Train (value). 
            If None, it defaults to "{:,.3g}".format.

        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        kwds = {"monitor"     : monitor,
                "ax"          : ax, 
                "colors"      : colors, 
                "plot_kwds"   : plot_kwds,
                "tight_layout": tight_layout, 
                "val_format"  : val_format}
        return plot_monitor(self, **kwds)

def InitializeWidget(verbose=True):
    
    '''Initialize progress widget'''
    if verbose:
        w1 = widgets.HTMLMath(value='Initializing . . .')
        w2 = widgets.HTMLMath(value='')
        display(widgets.HBox([w1, w2]))
        time.sleep(1)
        return w1, w2
    else: return None, None
    
def UpdateWidget(w, n_iter, rmse, loss):
    
    '''Update progress widget'''
    if (n_iter % 10)==0:
        w.value = ', '.join(('iter = {:,.0f}'.format(n_iter), 
                             'RMSE = {:,.4g}'.format(rmse),
                             'Loss = {:,.4g}'.format(loss)))
        
def ValidateTest(Train, Test):
    
    '''Validate X_test against X_train'''
    if Test is not None:
        if isinstance(Test, np.ndarray):
            if Train.shape != Test.shape:
                raise ValueError(f'`X_test` must be , {Train.shape}'
                                 f'Got {Test.shape} instead.') 
            else: return []
        else: raise ValueError(f'`X_test` must be numpy.ndarray, '
                               f'Got {type(Test)} instead.')
    else: return None
      
def plot_monitor(estimator, monitor="rmse", ax=None, colors=None, 
                 plot_kwds=None, tight_layout=True, val_format=None):
    
    '''
    Plot metrics from fitted `Ordinary_RSGD` or `RSGD_w_Bias`.
    
    Parameters
    ----------
    estimator : estimator object
        Fitted `Ordinary_RSGD` or `RSGD_w_Bias` estimator.
        
    monitor : {"loss", "rmse"}, default="rmse"
        Metric to be monitored.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, `ax` is created 
        with default figsize.

    colors : list of color-hex, default=None
        Number of color-hex must be equal to 2 i.e. Train and
        Test. If None, it uses default colors from Matplotlib.
        This overrides `plot_kwds`.
    
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to `ax.plot`.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
        
    val_format : string formatter, default=None
        String formatters (function) for minimum value of metric 
        of interest to be displayed on labels e.g. Train (value). 
        If None, it defaults to "{:,.3g}".format.
           
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(6, 4))[1]
        
    # Default : colors, val_format, and ylabel
    colors = ([ax._get_lines.get_next_color() for _ in range(2)] 
              if colors is None else colors)
    if val_format is None: val_format = "{:,.3g}".format
    ylabel = {"rmse" : "Root Mean Square\nError (RMSE)", 
              "loss" : "Mean Stochastic Gradient\nDescent (SGD) loss"}
        
    # Update keyword for ax.plot
    kwds = {"linewidth": 2.5}
    kwds = {**kwds, **({} if plot_kwds is None else plot_kwds)}
    kwds = [{**kwds, **{"color": colors[0]}},
            {**kwds, **{"color": colors[1]}}]
    data = [["train", "Train"], ["val", "Validate"]]
    
    for n in range(2):
        y = getattr(estimator.monitor, f"{monitor}_{data[n][0]}")
        if y is not None:
            if monitor == "loss": y = y/len(estimator.iters[n])
            x = np.arange(len(y)) + 1
            label = f"{data[n][1]} ({val_format(min(y))})"
            ax.plot(x, y, **{**kwds[n], **{"label": label}})
    
    ax.legend(loc="best", fontsize=12, framealpha=0)
    ax.set_xlabel(r"$n^{th}$ iteration", fontsize=12)
    ax.set_ylabel(ylabel[monitor], fontsize=12)
    ax.set_title("Learning Curves (Factor = {:,d}, Epoch = {:,d})"
                 .format(estimator.n_factors, estimator.n_iters), 
                 fontsize=14)
    if tight_layout: plt.tight_layout()
        
    return ax

def learning_rate(lr0=0.1, n_iters=100, k=None, drop_rate=10, 
                  method=None, min_lr=1e-6):
    
    '''
    `Learning_rate` adjusts the learning rate during training by 
    reducing the learning rate according to a pre-defined schedule. 
    
    Parameters
    ----------
    lr0 : float, default=0.1
        Initial learning rate.
    
    n_iters : int, default=100
        Number of iterations.
        
    k : float, default=None
        Decay factor is applicable when method is not None. The 
        default value is assigned as follows:
        - "time" : lr0 / n_iters
        - "step" : 0.9
        - "expo" : lr0 / n_iters

    drop_rate : int, default=10
        Number of iterations that learning rate is decreased. This 
        is relevant when method is "step".
        
    method : {"time", "step", "expo"}, default=None
        Algorithm of determining learning rates
        - "time" : Time-based decay, lr(t) = lr(t-1) / (1 + kt).
        - "step" : Step decay drops the learning rate by a factor 
                   every `drop_rate` (r), 
                   lr(t) = lr0 * k^floor((t + 1) / r).
        - "expo" : Exponential decay, lr(t) = lr0 * e^(−kt).
        If None, a constant learning rate (lr0) is applied.
        
    min_lr : float, default=1e-6
        Minimum learning rate. Any step whose learning rate is 
        below `min_lr`, is capped at this value.
        
    References
    ----------
    .. [1] https://towardsdatascience.com/learning-rate-schedules-
           and-adaptive-learning-rate-methods-for-deep-learning-
           2c8f433990d1
        
    Returns
    -------
    learning_rates : ndarray of float, of shape (n_iters,)
        An array of learning rates as per iteration.
    
    '''
    # Default value of `decay` (decay rate)
    default = {"time" : lr0/n_iters, "step" : 0.9, 
               "expo" : lr0/n_iters}
    if (not isinstance(k, numbers.Number)) & (method is not None): 
        k = default[method]
        
    if method=="time":     
        # lr(n) = lr(n-1) / (1 + n * decay)
        learning_rates, lr = [], lr0
        for t in np.arange(n_iters-1):
            lr *= 1 / float(1 + t * k)    
            learning_rates.append(lr)  
            
    elif method=="step":      
        # lr(n) = lr0 * decay^floor((n + 1) / drop_rate) 
        learning_rates = [lr0 * k**np.floor((t+1)/drop_rate) 
                          for t in np.arange(n_iters)] 
        
    elif method=="expo":              
        # lr(n) = lr0 * exp(-decay * n)
        learning_rates = [lr0 * np.exp(-k * t) 
                          for t in np.arange(n_iters)]  
        
    else: learning_rates = np.full(n_iters, lr0)
    return np.fmax(learning_rates, min_lr)

def rating_train_test(X, test_size=0.3, random_state=None):
    
    '''
    Split ratings into random train and test subsets
    
    Parameters
    ----------
    X : ndarray, of shape (n_users, n_items)
        User-item observation matrix.
            
    test_size : float, default=0.3
        It should be between 0.0 and 1.0 and represent the 
        proportion of the dataset to include in the test split. 
        
    random_state : int, default=None
        It controls the randomness of applying the split.
    
    Returns
    -------
    X_train, X_test : ndarray, of shape (n_users, n_items)
    
    '''
    # Initialize parameters
    r_index = ~np.isnan(X)
    r  = X[r_index].copy()
    nr = len(r)
    a  = np.arange(nr)
    
    # Randomize test index
    np.random.seed(random_state)
    kwds = dict(replace=False, size=int(test_size * nr))
    test = np.isin(a, np.random.choice(a, **kwds))
    
    # Train and Test datasets
    X_train, X_test  = X.copy(), X.copy()
    X_test[r_index]  = np.where( test, r, np.nan)
    X_train[r_index] = np.where(~test, r, np.nan)
    
    return X_train, X_test

def RecommenderScores(X_base, X_test, X_score, k=10, 
                      kwargs=None, ignore_nan=True):
    
    '''
    Calculate evaluation metrics for recommender.
    
    Parameters
    ----------
    X_base : ndarray, of shape (n_users, n_items)
        User-item observation matrix that is used as a reference 
        when compared to `X_test`. The ratings that get evaluated 
        are ones that are unknown in `X_base` but known in `X_test`.

    X_test : list or ndarray
        User-item observation matrix to be ranked and must 
        contain missing values or np.nan. If `X_test` is a list 
        (n_datasets,), each item must be User-item observation 
        matrix of shape (n_users, n_items).
        
    X_score : ndarray, of shape (n_users, n_items)
        An array of recommended scores. Score should be greater 
        than 0.
        
    k : int, default=5
        Number of top recommendations that is used to compute 
        "Hit Rate", "Precision", and "Recall".`k` is adjusted 
        according to available unknown ratings per user.
    
    kwargs : keywords, default=None
        Keyword arguments to be passed to "dcg_score", and 
        "ndcg_score"[2,3].
        
    ignore_nan : bool, defualt=True
        If True, `sample_weight` option is added to "kwargs". It 
        gives zero weight to user that has an empty row of ratings 
        (np.nan), otherwise 1. This affects when average score is 
        calculated i.e. dcg and ndcg. This overrides "kwargs".

    References
    ----------
    .. [1] https://machinelearningmedium.com/2017/07/24/discounted-
           cumulative-gain/
    .. [2] https://scikit-learn.org/0.24/modules/generated/sklearn.
           metrics.dcg_score.html
    .. [3] https://scikit-learn.org/0.24/modules/generated/sklearn.
           metrics.ndcg_score.html#sklearn.metrics.ndcg_score
    .. [4] https://medium.com/fnplus/evaluating-recommender-systems
           -with-python-code-ae0c370c90be
           
    Returns
    -------
    scores : collections.namedtuple
    
        [1] rmse : ndarray, of shape (n_datasets,)
            Root mean square error.
            
        [2] mae : ndarray of shape (n_datasets,)
            Mean absolute error.
    
        [3] dcg : ndarray, of shape (n_datasets,)
            Mean Discounted Cumulative Gain (DCG) measures the 
            gain, of sorted items wrt. their positionin `X_score`. 
            The gain is accumulated from the top to the bottom 
            from the sorted items with the gain of each position 
            discounted [1,2,3]. 

        [4] ndcg : ndarray, of shape (n_datasets,)
            Mean Normalized Discounted Cumulative Gain (NDCG). DCG 
            is divided by IDCG (Ideal DCG), which is a DCG that 
            obtained when `X_test` and `X_score` are perfectly 
            correlated.
            
        [5] hitrate : ndarray, of shape (n_datasets,)
            If one of `k` top recommendations is selected per user, 
            it is consiered as a hit. The `hitrate` is the total 
            number of hits divided by number of users.
      
        [6] precision : ndarray, of shape (n_datasets,)
            Average precision given `k` top recommendations.
        
        [7] recall : ndarray, of shape (n_datasets,)
            Average recall given `k` top recommendations.
            
        [8] n_ratings : ndarray, of shape (n_datasets,)
            Number of new ratings compared to `X_base`.
            
    '''
    # Initialize parameters
    if not isinstance(X_test, list): X_test = [X_test]
    n_datasets = len(X_test)
    keys = ["rmse", "mae", "dcg", "ndcg", "hitrate", 
            "precision", "recall", "n_ratings"]                                      
    data = dict((key, np.full(n_datasets, np.nan)) for key in keys)

    # Default keywords for `dcg_score`, and `ndcg_score`
    if (kwargs is None) | (not isinstance(kwargs, dict)):
        kwargs = dict(k=None, sample_weight=None)
    
    # Extract only predicted scores (unknown in `X_base`)  
    # and replace known scores with zeros.
    X_pred = np.where(np.isnan(X_base), X_score, 0)
    
    # Determine position of `k` top recommendations.
    n_users, n_items = X_base.shape
    ranks = n_items - np.argsort(np.isnan(X_base)*X_score, 1)
    top_k_ranks = np.where((ranks <= k), 1, 0)
    
    for (n, X_true) in enumerate(X_test):
        
        # New ratings (scores)
        index  = np.isnan(X_base) & (~np.isnan(X_true))
        X_true = np.where(index, X_true , 0)
        
        # Adjust "sample_weight".
        if ignore_nan: sample_weight = X_true.sum(axis=1)>0
        else: sample_weight = np.ones(n_users)
        kwargs.update({"sample_weight" : sample_weight})
        
        # [1] Root Mean Square Error (RMSE)
        X = np.where(X_true==0, np.nan, X_true)
        data["rmse"][n] = np.sqrt(np.nanmean((X - X_pred)**2))
        
        # [2] Mean Absolute Error (MAE)
        data["mae"][n] = np.nanmean(abs(X - X_pred))
        
        # [3] Mean Discounted Cumulative Gain
        data["dcg"][n] = dcg_score(X_true, X_pred, **kwargs)
        
        # [4] Mean Normalized Discounted Cumulative Gain
        data["ndcg"][n] = ndcg_score(X_true, X_pred, **kwargs)
        
        # [5] Hit rate (given `k` top recommendations)        
        n_hits = ((X_true * top_k_ranks).sum(axis=1)>0)
        data["hitrate"][n] = n_hits.sum()/n_users
        
        # [6] Precision (given `k` top recommendations)
        # `k` cannot exceed unknown ratings in `X_base` as per user
        max_items = np.isnan(X_base).sum(axis=1) 
        data["precision"][n] = (n_hits/np.fmin(max_items, k)).mean()
        
        # [7] Recall (given `k` top recommendations)
        n_ratings = (X_true>0).sum(axis=1)
        n_ratings = np.where(n_ratings==0, 1, n_ratings)
        data["recall"][n] = (n_hits/n_ratings).mean()
        
        # [8] Number of new ratings
        data["n_ratings"][n] = (X_true>0).sum(axis=1).sum()
    
    scores = collections.namedtuple('scores', keys)    
    scores = scores(**data)
    
    return scores

def ItemReccommender(X, scores, items=None, max_item=None, min_score=None):
    
    '''
    Recommendation of items.
    
    Parameters
    ----------
    X : ndarray, of shape (n_users, n_items)
        Ratings of items from users. "X" must contain missing 
        values (np.nan) because algorithm only recommends item 
        where missing is found.
        
    scores : ndarray, of shape (n_users, n_items)
        Recommended ratings (scores) of items from each user.
    
    items : list of str, default=None
        A list of items. If None, items default to [0,1,...,n] 
        where n is the nth index wrt. columns in "X".
    
    max_item : int, default=None
        Max number of recommended items. If None, it defaults 
        to total number of items, X.shape[1].
        
    min_score : float, default=None
        Min acceptable score. If None, it defaults to minimum 
        score, min(scores).
    
    Returns
    -------
    rec_items : ndarray of shape (n_users, n_items) 
        The order of recommended items corresponds to that in 
        "rec_scores". If "items" is not provided, index wrt. to
        columns in "X" e.g. [0,1,...,n], is used instead.
    
    rec_scores : ndarray of shape (n_users, n_items) 
        Scores in each row are monotonically ordered from highest 
        to lowest. The number of recommended items depends on 2
        parameters i.e. "max_item", and "min_score".
   
    '''
    # Default product names
    if items is None: items = np.arange(X.shape[1])
    try: items = np.array(items).astype(float)
    except: items = np.array(items).astype(object)
    
    # Maximum number of recommendations.
    if max_item is None: max_item = X.shape[1]
    max_item = max(min(X.shape[1], max_item),1)
    
    # Minimum acceptable scores.
    if min_score is None: min_score = np.min(scores)
    min_score = min(np.max(scores), min_score)
    
    if np.sum(np.isnan(X))==0:
        raise ValueError(f'`X` must contain missing values '
                         f'(np.nan) because algorithm only '
                         f'recommends item where missing is found') 
        
    # Only evaluate the unknown scores (np.nan) by 
    # converting scores that exist in X to min_score - 1.
    scores_ = np.array(scores).copy()
    scores_[~np.isnan(X)] = min_score - 1
    
    rec_scores, rec_items = [], []
    for n in np.arange(X.shape[0]):
    
        index = np.argsort(scores_[n, :])
        sorted_scores= scores_[n, index][::-1].copy() 
        sorted_items = items[index][::-1].copy()
        
        n_items = min(sum(sorted_scores>=min_score), max_item)
        sorted_scores[n_items:] = np.nan
        sorted_items[n_items:] = np.nan
        
        rec_scores+= [sorted_scores]
        rec_items += [sorted_items]

    return np.array(rec_items), np.array(rec_scores)