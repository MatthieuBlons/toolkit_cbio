from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt


def survival_curve(
    df, metric, event, stratif=None, time_pt=None, time_unit="days", ax=None, show=True
):

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

    def plot_km_curve(time, survival_prob, conf_int, label):
        ax.step(time, survival_prob, label=label, linewidth=1.5, where="post")
        ax.fill_between(time, conf_int[0], conf_int[1], alpha=0.2, step="post")
        ax.set_ylim(bottom=0, top=1)

    tmp = df.dropna(axis=0, how="any", subset=[metric, event], inplace=False)
    if stratif is not None:  # Stratified survival curves
        # if stratif is list
        # new_var = results_table.apply(make_new_var(on_vars), axis=1)
        # stratif = "_".join(on_vars)
        # df[stratif] = new_var
        tmp = tmp.dropna(axis=0, subset=stratif, inplace=False)
        for group in tmp[stratif].unique():
            mask = tmp.loc[tmp[stratif] == group]
            time, survival_prob, conf_int = kaplan_meier_estimator(
                mask[event].astype("bool"),
                mask[metric],
                conf_level=0.95,
                conf_type="log-log",
            )
            mask = [survival_prob[i] != 0 for i in range(len(survival_prob))]
            plot_km_curve(
                time[mask], survival_prob[mask], conf_int[:, mask], f"{stratif} {group}"
            )

        ax.set_title(f"Kaplan Meier Estimates for {metric} with respect to {stratif}")

    else:  # Single survival curve
        time, survival_prob, conf_int = kaplan_meier_estimator(
            tmp[event].astype("bool"), tmp[metric], conf_level=0.95, conf_type="log-log"
        )
        plot_km_curve(time, survival_prob, conf_int, metric)
        ax.set_title(f"{metric}")
    if time_pt:
        ax.vlines(
            x=time_pt,
            colors="red",
            linestyles="dashed",
            ymin=0,
            ymax=1,
            label=f"{time_pt} {time_unit}",
        )
    ax.set_xlabel(f"{time_unit}")
    ax.set_ylabel("Fraction of patients")
    ax.legend()

    if show:
        plt.show()

    return tmp
