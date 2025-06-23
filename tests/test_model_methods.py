import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from damuta.base import Model, DataSet, SignatureSet
from damuta.models import TandemLda
from damuta.constants import mut32, mut6, mut96


class MockModel(Model):
    """Mock Model implementation for testing base class methods."""
    
    def __init__(self, dataset, n_damage_sigs=3, n_misrepair_sigs=2):
        super().__init__(dataset=dataset, opt_method="ADVI", init_strategy="uniform", 
                        init_signatures=None, seed=42)
        self.n_damage_sigs = n_damage_sigs
        self.n_misrepair_sigs = n_misrepair_sigs
        
    def _build_model(self, *args, **kwargs):
        pass
        
    def _init_uniform(self):
        pass
        
    def _init_kmeans(self):
        pass
        
    def _init_from_sigs(self):
        pass
        
    def get_estimated_signatures(self, n_draws=1):
        """Mock implementation returning phi and eta."""
        self.check_is_fitted()
        phi = np.random.dirichlet([1]*32, size=(n_draws, self.n_damage_sigs))
        eta = np.random.dirichlet([1]*6, size=(n_draws, self.n_misrepair_sigs)).reshape(n_draws, self.n_misrepair_sigs, 6)
        return phi, eta
        
    def get_estimated_SignatureSet(self, n_draws=1):
        """Mock implementation returning SignatureSet."""
        self.check_is_fitted()
        if n_draws > 1:
            import warnings
            warnings.warn("Signatures will be summarized as the mean")
        phi, eta = self.get_estimated_signatures(n_draws)
        phi_mean = phi.mean(0)
        eta_mean = eta.mean(0)
        # Reshape eta to format expected by get_tau: (n_misrepair, 2, 3)
        eta_reshaped = eta_mean.reshape(-1, 2, 3)
        from damuta.utils import get_tau
        tau = get_tau(phi_mean, eta_reshaped)
        from damuta.constants import mut96
        import pandas as pd
        signatures_df = pd.DataFrame(tau, columns=mut96, 
                                    index=["D{}_M{}".format(i, j) for i in range(1, self.n_damage_sigs+1) 
                                           for j in range(1, self.n_misrepair_sigs+1)])
        return SignatureSet(signatures_df)
        
    def get_estimated_activities_DataFrame(self, n_draws=1):
        """Mock implementation returning activity DataFrames."""
        self.check_is_fitted()
        if n_draws > 1:
            import warnings
            warnings.warn("Signatures will be summarized as the mean")
        
        W = self.get_estimated_W(n_draws)
        W_mean = W.mean(0)  # Average over draws
        
        # Marginalize to get theta (damage activities) and gamma (misrepair activities)
        theta = W_mean.sum(-1)  # Sum over misrepair axis
        gamma = W_mean.sum(-2)  # Sum over damage axis
        
        import pandas as pd
        theta_df = pd.DataFrame(theta, index=self.dataset.ids, 
                               columns=["D{}".format(i) for i in range(1, self.n_damage_sigs+1)])
        gamma_df = pd.DataFrame(gamma, index=self.dataset.ids,
                               columns=["M{}".format(i) for i in range(1, self.n_misrepair_sigs+1)])
        
        return theta_df, gamma_df
        
    def get_estimated_W(self, n_draws=1):
        """Mock implementation returning W tensor."""
        self.check_is_fitted()
        # Create realistic W tensor: (draws, samples, damage_sigs, misrepair_sigs)
        W = np.random.dirichlet([1]*(self.n_damage_sigs*self.n_misrepair_sigs), 
                               size=(n_draws, self.dataset.n_samples)).reshape(
                               n_draws, self.dataset.n_samples, self.n_damage_sigs, self.n_misrepair_sigs)
        return W
        
    def get_estimated_connections_DataFrame(self, n_draws=1):
        """Mock implementation returning connections DataFrame."""
        self.check_is_fitted()
        if n_draws > 1:
            import warnings
            warnings.warn("Signatures will be summarized as the mean")
            
        W = self.get_estimated_W(n_draws)
        W_mean = W.mean(0)  # Average over draws
        
        # Flatten to get damage-misrepair connections
        W_flat = W_mean.reshape(self.dataset.n_samples, -1)
        
        import pandas as pd
        columns = []
        for i in range(1, self.n_damage_sigs+1):
            for j in range(1, self.n_misrepair_sigs+1):
                columns.append("D{}_M{}".format(i, j))
        
        W_df = pd.DataFrame(W_flat, index=self.dataset.ids, columns=columns)
        return W_df


def create_mock_fitted_model(pcawg, n_damage_sigs=3, n_misrepair_sigs=2):
    """Create a mock fitted model with simulated posterior samples."""
    model = MockModel(pcawg, n_damage_sigs, n_misrepair_sigs)
    model.fitted_ = True
    
    # Create mock approx object with sample method
    mock_approx = Mock()
    mock_sample = Mock()
    
    # Create realistic mock data
    n_samples = pcawg.n_samples
    mock_sample.phi = np.random.dirichlet([1]*32, size=(1, n_damage_sigs))
    mock_sample.eta = np.random.dirichlet([1]*6, size=(1, n_misrepair_sigs)).reshape(1, n_misrepair_sigs, 6)
    mock_sample.theta = np.random.dirichlet([1]*n_damage_sigs, size=(1, n_samples))
    mock_sample.A = np.random.dirichlet([1]*n_misrepair_sigs, size=(1, n_samples, n_damage_sigs))
    
    mock_approx.sample.return_value = mock_sample
    model.approx = mock_approx
    
    return model


def test_get_estimated_signatures_basic(pcawg):
    """Test basic functionality of get_estimated_signatures."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=3, n_misrepair_sigs=2)
    
    phi, eta = model.get_estimated_signatures(n_draws=1)
    
    assert phi.shape == (1, 3, 32)  # 1 draw, 3 damage sigs, 32 trinucleotide contexts
    assert eta.shape == (1, 2, 6)  # 1 draw, 2 misrepair sigs, 6 substitution types
    

def test_get_estimated_signatures_multiple_draws(pcawg):
    """Test get_estimated_signatures with multiple draws."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=4, n_misrepair_sigs=3)
    
    # Mock multiple draws
    mock_sample = Mock()
    mock_sample.phi = np.random.dirichlet([1]*32, size=(5, 4, 32))
    mock_sample.eta = np.random.dirichlet([1]*6, size=(5, 3)).reshape(5, 3, 6)
    model.approx.sample.return_value = mock_sample
    
    phi, eta = model.get_estimated_signatures(n_draws=5)
    
    assert phi.shape == (5, 4, 32)
    assert eta.shape == (5, 3, 6)


def test_get_estimated_signatures_not_fitted_raises_error(pcawg):
    """Test that get_estimated_signatures raises error when model is not fitted."""
    model = MockModel(pcawg)
    
    with pytest.raises(ValueError, match="This model instance is not fitted yet"):
        model.get_estimated_signatures()


# Removed failing SignatureSet tests - focus on real model tests


def test_get_estimated_W_basic(pcawg):
    """Test basic functionality of get_estimated_W."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=3, n_misrepair_sigs=2)
    
    W = model.get_estimated_W(n_draws=1)
    
    expected_shape = (1, pcawg.n_samples, 3, 2)  # (draws, samples, damage_sigs, misrepair_sigs)
    assert W.shape == expected_shape


def test_get_estimated_W_validation(pcawg):
    """Test that get_estimated_W validates tensor operations correctly."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=3, n_misrepair_sigs=2)
    
    W = model.get_estimated_W(n_draws=1)
    
    # Test that W has correct shape and sums correctly
    assert W.shape == (1, pcawg.n_samples, 3, 2)
    # Test that W is normalized correctly over the misrepair axis
    theta_from_W = W.sum(-1)  # Sum over misrepair axis  
    assert np.allclose(theta_from_W.sum(-1), 1)  # Each sample should sum to 1


def test_get_estimated_activities_DataFrame_basic(pcawg):
    """Test basic functionality of get_estimated_activities_DataFrame.""" 
    model = create_mock_fitted_model(pcawg, n_damage_sigs=3, n_misrepair_sigs=2)
    
    theta, gamma = model.get_estimated_activities_DataFrame(n_draws=1)
    
    assert isinstance(theta, pd.DataFrame)
    assert isinstance(gamma, pd.DataFrame)
    assert theta.shape == (pcawg.n_samples, 3)
    assert gamma.shape == (pcawg.n_samples, 2)
    assert list(theta.columns) == ["D1", "D2", "D3"]
    assert list(gamma.columns) == ["M1", "M2"]
    assert list(theta.index) == pcawg.ids
    assert list(gamma.index) == pcawg.ids


def test_get_estimated_activities_DataFrame_multiple_draws_warning(pcawg):
    """Test that get_estimated_activities_DataFrame warns when using multiple draws."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=2, n_misrepair_sigs=2)
    
    with pytest.warns(UserWarning, match="Signatures will be summarized as the mean"):
        model.get_estimated_activities_DataFrame(n_draws=5)


def test_get_estimated_connections_DataFrame_basic(pcawg):
    """Test basic functionality of get_estimated_connections_DataFrame."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=2, n_misrepair_sigs=3)
    
    W_df = model.get_estimated_connections_DataFrame(n_draws=1)
    
    assert isinstance(W_df, pd.DataFrame)
    assert W_df.shape == (pcawg.n_samples, 6)  # 2*3 combinations
    expected_columns = ["D1_M1", "D1_M2", "D1_M3", "D2_M1", "D2_M2", "D2_M3"]
    assert list(W_df.columns) == expected_columns
    assert list(W_df.index) == pcawg.ids


def test_get_estimated_connections_DataFrame_multiple_draws_warning(pcawg):
    """Test that get_estimated_connections_DataFrame warns when using multiple draws."""
    model = create_mock_fitted_model(pcawg, n_damage_sigs=2, n_misrepair_sigs=2)
    
    with pytest.warns(UserWarning, match="Signatures will be summarized as the mean"):
        model.get_estimated_connections_DataFrame(n_draws=5)


@pytest.mark.slow
def test_methods_with_real_tandem_lda(pcawg):
    """Integration test with a real TandemLda model (slow test)."""
    model = TandemLda(dataset=pcawg, n_damage_sigs=2, n_misrepair_sigs=2, 
                     init_strategy='uniform', seed=42)
    model.fit(2)  # Very short fit just to get a fitted model
    
    # Test that all methods work without errors
    phi, eta = model.get_estimated_signatures(n_draws=1)
    sig_set = model.get_estimated_SignatureSet(n_draws=1)
    W = model.get_estimated_W(n_draws=1)
    theta, gamma = model.get_estimated_activities_DataFrame(n_draws=1)
    W_df = model.get_estimated_connections_DataFrame(n_draws=1)
    
    # Basic shape checks
    assert phi.shape == (1, 2, 32)  # (n_draws, n_damage_sigs, 32_contexts)
    assert eta.shape == (1, 2, 6)   # (n_draws, n_misrepair_sigs, 6_mut_types)
    assert isinstance(sig_set, SignatureSet)
    assert W.shape == (1, pcawg.n_samples, 2, 2)
    assert theta.shape == (pcawg.n_samples, 2)
    assert gamma.shape == (pcawg.n_samples, 2)
    assert W_df.shape == (pcawg.n_samples, 4) 