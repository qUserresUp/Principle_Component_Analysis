import numpy as np
'''
Homework5: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_samples, n_features].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()


    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        need self.x_bar
        '''

        n_samples, n_features = self.X.shape
        X_np = np.array(self.X)
        X_np = X_np.transpose()

        One_n = np.ones(n_samples,)
        self.X_bar = (1/n_samples) * np.inner(X_np, One_n)
        X_bar_x = (self.X_bar).shape
        """
        print("X_bar shape: ", X_bar_x ) 
        
        256
        """

        centered_X = X_np - np.outer(self.X_bar, One_n)
        u, s, vh = np.linalg.svd(centered_X)
        g = u[:, :self.n_components]

        """z here has shape [k*n]"""
        z = np.dot(np.transpose(g), centered_X)

        return g, z.transpose()



    def get_reduced(self, X=None):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_any, n_features] or None.
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_any, n_components].
        '''

        if X is None:
            return self.Xp
        else:
            return X@self.Up


    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        Return:
        X_re: The reconstructed matrix of shape [n_samples, n_features].
        '''

        z = np.transpose(Xp)
        X_rec = np.dot(self.Up, z)
        return np.transpose(X_rec)



def reconstruct_error(A, B):
    '''
    To compute the reconstruction error.

    Args:
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return:
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''

    return np.linalg.norm((A - B), 'fro')








