
import os
import tqdm
import torch
import numpy as np
import pandas as pd

from utils import nn_utils
import torch.nn.functional as F

from sklearn.metrics import classification_report


def get_top1_error(y_true, y_pred, target_names):

    if target_names:
        top1_error_dict = {k: 0 for k in target_names}
    else:
        top1_error_dict = {k: 0 for k in list(set(y_true))}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            if target_names:
                top1_error_dict[target_names[true_label]] += 1
            else:
                top1_error_dict[true_label] += 1

    df_top1_error = pd.DataFrame([top1_error_dict]).T
    df_top1_error.rename(columns={0: 'Error'}, inplace=True)
    total_error = df_top1_error['Error'].sum()
    print('Top-1 error:')
    print(df_top1_error)
    print(f'Total top-1 error: {total_error}')
    return total_error


def get_classification_report(y_true, y_pred, target_names):
    out_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    out_txt = classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
    return out_dict, out_txt


def eval_epoch(weights_path, evaluate_dict, plot_roc_curve=False):
    model = evaluate_dict['model']
    criterion = evaluate_dict['criterion']
    test_loader = evaluate_dict['test_loader']
    class_dict = evaluate_dict['class_dict']
    target_names = evaluate_dict['target_names']
    device = evaluate_dict['device']
    data_hash = evaluate_dict['data_hash']
    model.cuda()

    model.load_state_dict(torch.load(weights_path))
    model.eval()
    y_true, y_pred = [], []

    if plot_roc_curve:
        logits_array = None

    with torch.no_grad():
        eval_loss = 0.0
        for i, test_batch in enumerate(test_loader):
            # Get validation batch
            inputs, labels = test_batch['image'], test_batch['class']
            y_true.extend(labels.cpu().numpy().tolist())
            # Send data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            eval_loss += criterion(outputs, labels)
            # Collect predictions
            _, indicies = torch.max(outputs, 1)
            y_pred.extend(indicies.cpu().numpy().tolist())

            # Collect logits array for AUC
            if plot_roc_curve:
                outputs_proba = F.softmax(outputs, dim=1)
                if isinstance(logits_array, np.ndarray):
                    logits_array = np.vstack([logits_array, outputs_proba.cpu().numpy()])
                else:
                    logits_array = outputs_proba.cpu().numpy()

    report_dict, report = get_classification_report(y_true, y_pred, target_names)
    top1_error = get_top1_error(y_true, y_pred, target_names)
    report_dict.update({'top1_error': top1_error})

    if plot_roc_curve:
        fpr, tpr, roc_auc = nn_utils.compute_roc_auc(y_true, logits_array, class_dict)
        print('ROC AUC'.center(50, '='))
        print(''.join(list(f'{k}: {v}\n' for k, v in roc_auc.items())))
        nn_utils.plot_multiclass_roc_curves(fpr, tpr, roc_auc, class_dict, True, data_hash)

    print(f'Test loss: {eval_loss}')
    print(report)

    return report_dict


def eval_epoch_on_external_data(weights_path, evaluate_dict):

    """
    Оценка модели на внешних данных.
    Важно, чтобы в словаре trained_data_class_dict был словарь с данными, на которых проходило обучение,
    а в словаре external_data_class_dict - словарь с данными, отобранными извне.
    :param weights_path: string
    :param evaluate_dict: dict
    :return:
    """

    model = evaluate_dict['model']
    criterion = evaluate_dict['criterion']
    test_loader = evaluate_dict['test_loader']
    external_data_class_dict = evaluate_dict['external_data_class_dict']   # string: digit
    trained_data_class_dict = evaluate_dict['trained_data_class_dict']     # string: digit
    device = evaluate_dict['device']
    data_hash = evaluate_dict['data_hash']
    external_invert_class_dict = {v: k for k, v in external_data_class_dict.items()}   # digit: string
    trained_invert_class_dict = {v: k for k, v in trained_data_class_dict.items()}     # digit: string
    model.cuda()

    model.load_state_dict(torch.load(weights_path))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        eval_loss = 0.0
        for i, test_batch in enumerate(test_loader):
            # Get validation batch
            inputs, labels = test_batch['image'], test_batch['class']
            y_true.extend([external_invert_class_dict[x] for x in labels.cpu().numpy().tolist()] )
            # Send data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            eval_loss += criterion(outputs, labels)
            # Collect predictions
            _, indicies = torch.max(outputs, 1)
            y_pred.extend([trained_invert_class_dict[x] for x in indicies.cpu().numpy().tolist()])

    report_dict, report = get_classification_report(y_true, y_pred, None)
    top1_error = get_top1_error(y_true, y_pred, None)
    report_dict.update({'top1_error': top1_error})
    print(f'Test loss: {eval_loss}')
    print(report)
    return report_dict


def model_evaluate(evaluate_dict):

    weights_folder = evaluate_dict['weights_folder']
    weights_list = [os.path.join(weights_folder, x) for x in os.listdir(weights_folder) if '.pth' in x]
    epoch_dict = dict([(int(x.split('.')[0].split('_')[-1]), x) for x in weights_list])
    sorted_epochs = sorted(epoch_dict)

    reports_list = []
    for epoch in tqdm.tqdm(sorted_epochs):
        weights_path = epoch_dict[epoch]

        print('Epoch: ', epoch)
        print('Weights path: ', weights_path)

        if evaluate_dict['external_data_class_dict']:
            report_dict = eval_epoch_on_external_data(weights_path, evaluate_dict)
        else:
            report_dict = eval_epoch(weights_path, evaluate_dict)

        report_dict.update({'epoch': epoch})
        reports_list.append(report_dict)
    return reports_list


