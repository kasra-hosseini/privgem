from privgem import tabular_patectgan
from joblib import dump

def train_save_pate_models(data, 
                           discrete_columns,
                           epsilon, 
                           batch_size, 
                           noise_multiplier, 
                           moments_order,
                           output_save_path, 
                           device,
                           model_save_path):
    
    pate_model = tabular_patectgan(verbose=True, 
                                   epsilon=epsilon, 
                                   batch_size=batch_size, 
                                   noise_multiplier=noise_multiplier, 
                                   moments_order=moments_order, 
                                   output_save_path=output_save_path,
                                   device=device)
    pate_model.train(data, discrete_columns)
    dump(pate_model, model_save_path)