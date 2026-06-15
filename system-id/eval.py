import torch
from metrics import summarize_predictions


@torch.no_grad()
def eval_model(model, loader, device, analytic=False):
    preds = []
    trues = []
    Ts = []
    Ac = []
    ids = []

    for batch in loader:
        A_locals = batch['A_locals'].to(device)
        A_true = batch['A_true'].to(device)
        T = batch['T'].to(device)
        A_canon = batch['A_canon'].to(device)
        system_idx = batch['system_idx']

        if analytic:
            A_pred = A_locals.mean(dim=1)
        else:
            A_pred = model(A_locals)

        preds.append(A_pred.cpu())
        trues.append(A_true.cpu())
        Ts.append(T.cpu())
        Ac.append(A_canon.cpu())
        ids.append(system_idx.cpu() if torch.is_tensor(system_idx) else torch.tensor(system_idx))

    A_pred = torch.cat(preds, dim=0)
    A_true = torch.cat(trues, dim=0)
    T = torch.cat(Ts, dim=0)
    A_canon = torch.cat(Ac, dim=0)
    system_idx = torch.cat(ids, dim=0)

    out = summarize_predictions(A_pred, A_true, A_canon=A_canon, T=T)
    out['A_pred'] = A_pred
    out['A_true'] = A_true
    out['T'] = T
    out['A_canon'] = A_canon
    out['system_idx'] = system_idx
    return out
