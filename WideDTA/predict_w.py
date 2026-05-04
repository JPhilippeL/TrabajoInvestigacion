import data_w
from data_w import widedata
from data_w import*
# import train_w
# from train_w import*
trained_wmodel=torch.load('wide.pt')
dataset = widedata(ligand_path, protein_path,keys,motif_path,affinity_path)
train_loader, test_loader = load_splitset(dataset, .2)


from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
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

def predict_and_evaluate_w(model_,test):
    out21 = []
    count = 0

    actual_values = [] ### Added to calculate MSE and CI

    for i,j in(test):
        m = i[0]
        #m = m.reshape((1, 10, 10)) # For KIBA dataset
        m = m.reshape((1, 141, 101)) # For Davis dataset
        p = i[1]
        #p = p.reshape((1, 6729, 594)) # For KIBA dataset
        p = p.reshape((1, 9552, 467)) # For Davis dataset
        mt = i[2]
        #mt = mt.reshape((1, 1076, 32)) # For KIBA dataset
        mt = mt.reshape((1, 2017, 96)) # For Davis dataset
        predict = model_(p, m, mt)
        out21.append(predict)

        actual_values.append(j) #### Added to calculate MSE and CI

        count += 1
        if count == 50:
            break

    ### Added to calculate MSE and CI
    # If out2 is a list of tensors, concatenate it
    #if isinstance(out21[0], torch.Tensor):
    #    out21 = torch.cat(out21)
    
    actual_values = torch.cat(actual_values) # Convert list of tensors to a single tensor
    mse, ci = calculate_metrics(out21, actual_values)

    return out21, mse, ci

#pred=predict_w(trained_wmodel,test_loader)
#if __name__ == '__main__':
#    print(pred)

### Added to calculate MSE and CI
pred, mse, ci = predict_and_evaluate_w(trained_wmodel, test_loader)
#print("Predictions: ", pred)
print("MSE: ", mse)
print("CI: ", ci)