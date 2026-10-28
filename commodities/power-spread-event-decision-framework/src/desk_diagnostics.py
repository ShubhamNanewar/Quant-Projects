"""Post-hoc diagnostics of frozen forecasts; no parameter fitting or trade P&L."""
import numpy as np
import pandas as pd


def validate(frame):
    required = {"datetime_utc", "model", "spread", "prediction", "threshold", "selected", "direction"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing prediction fields")
    out = frame.copy()
    out["datetime_utc"] = pd.to_datetime(out.datetime_utc, utc=True)
    if out.duplicated(["datetime_utc", "model"]).any():
        raise ValueError("Duplicate model/hour")
    if not np.isfinite(out[["spread", "prediction", "threshold", "direction"]].to_numpy()).all():
        raise ValueError("Non-finite inputs")
    if not out.selected.isin([True, False]).all():
        raise ValueError("Selection flags must be boolean")
    if not np.array_equal(out.selected.to_numpy(), (out.prediction.abs() >= out.threshold).to_numpy()):
        raise ValueError("Selection must match frozen threshold")
    if not np.array_equal(out.direction.to_numpy(), np.sign(out.prediction).to_numpy()):
        raise ValueError("Direction mismatch")
    local = out.datetime_utc.dt.tz_convert("America/Los_Angeles")
    out["delivery_hour"] = local.dt.hour
    out["season"] = local.dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
    out["error"] = (out.spread-out.prediction).abs()
    out["signed_spread"] = out.direction*out.spread
    return out


def summarize(frame, keys):
    rows=[]
    for key, group in frame.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        selected=group.loc[group.selected, "signed_spread"]
        n=len(selected)
        rows.append(dict(zip(keys,key), hours=len(group), mae=group.error.mean(),
                         selected_hours=n, selection_fraction=n/len(group),
                         positive_fraction=(selected>0).mean() if n else np.nan,
                         mean_signed_spread=selected.mean(), worst_signed_spread=selected.min(),
                         lower_five_percent_mean=selected.nsmallest(max(1,int(np.ceil(.05*n)))).mean() if n else np.nan))
    return pd.DataFrame(rows)


def cost_sensitivity(frame, hurdles=(0,1,2,5,10)):
    rows=[]
    for model, group in frame.groupby("model"):
        values=group.loc[group.selected,"signed_spread"]
        for hurdle in hurdles:
            rows.append(dict(model=model, hypothetical_hurdle=hurdle, selected_hours=len(values),
                             adjusted_mean=(values-hurdle).mean(), positive_fraction=((values-hurdle)>0).mean()))
    return pd.DataFrame(rows)
