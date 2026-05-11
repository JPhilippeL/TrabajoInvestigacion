from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import data
from data import*

### Function added to calculate MSE and CI
def calculate_metrics(predictions, actuals):
    # Check if predictions is a list of tensors or a single tensor
    if isinstance(predictions, list):
        predictions = torch.cat(predictions)
    
    predictions_np = predictions.detach().numpy()
    actuals_np = actuals.numpy()

    mse = mean_squared_error(actuals_np, predictions_np)
    ci = concordance_index(actuals_np, predictions_np)
    
    return mse, ci

def predict_and_evaluate(model, test_loader):
    out2 = []
    count = 0

    actual_values = [] ### Added to calculate MSE and CI
    
    for i, j in test_loader:
        m1 = i[0]
        m1 = m1.reshape((1, 62, 50))
        p1 = i[1]
        p1 = p1.reshape((1, 25, 600))
        
        predict = model(m1, p1)
        out2.append(predict)

        actual_values.append(j) #### Added to calculate MSE and CI
        
        count += 1
        if count == 50:
            break
    
    ### Added to calculate MSE and CI
    # If out2 is a list of tensors, concatenate it
    if isinstance(out2[0], torch.Tensor):
        out2 = torch.cat(out2)
    
    actual_values = torch.cat(actual_values) # Convert list of tensors to a single tensor
    mse, ci = calculate_metrics(out2, actual_values)
    
    return out2, mse, ci


trained_model = torch.load('deep.pt')
dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
train_loader, test_loader = load_splitset(dataset, .1)
### Added to calculate MSE and CI
pred, mse, ci = predict_and_evaluate(trained_model, test_loader)
print("Predictions: ", pred)
print("MSE: ", mse)
print("CI: ", ci)
