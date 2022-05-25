"""Generates CSV for vehicle distributions."""
import os
import pandas as pd
import random

from flow.data_pipeline.data_pipeline import upload_to_s3


def generate_distribution(models, proportions=None, seed=None):
    """
    Generate a distribution of energy models.

    Parameters
    ----------
    models : list(str) | str
        The names of the energy model(s) included in this distribution, matching names in query.py
    proportions : list(int) | list(float)
        The relative weights used to generate the distribution. If None is provided, the models will be
        chosen with equal weights. The proportions do *not* need to sum up to any particular number. The
        only restriction is that it can not sum up to zero.
    seed : int
        The random seed used for this distribution, for reproducibility purposes.

    Returns
    -------
    list of models
    """
    if type(models) != list:
        models = [models]
    random.seed(seed)
    return random.choices(
        population=models,
        weights=proportions,
        k=5000)


if __name__ == "__main__":
    data = dict()
    data['all_tacoma_fit'] = generate_distribution('TACOMA_FIT_DENOISED_ACCEL')
    data['all_prius_ev_fit'] = generate_distribution('PRIUS_FIT_DENOISED_ACCEL')
    data['fifty_fifty_tacoma_prius'] = generate_distribution(
        models=['TACOMA_FIT_DENOISED_ACCEL', 'PRIUS_FIT_DENOISED_ACCEL'],
        seed=817175)
    data['all_midsize_sedan'] = generate_distribution('MIDSIZE_SEDAN_FIT_DENOISED_ACCEL')
    data['all_midsize_suv'] = generate_distribution('MIDSIZE_SUV_FIT_DENOISED_ACCEL')
    data['distribution_v0'] = generate_distribution(
        models=['COMPACT_SEDAN_FIT_DENOISED_ACCEL',  # compact sedan 18.96% + prius 1.42%
                'MIDSIZE_SEDAN_FIT_DENOISED_ACCEL',  # midsize sedan 28.44%
                'MIDSIZE_SUV_FIT_DENOISED_ACCEL',  # midsize suv 6.26% + rav4 8.91%
                'LIGHT_DUTY_PICKUP',  # light duty pickup 2.28% + tacoma 6.64%
                'CLASS3_PND_TRUCK'],  # class3 pnd truck 13.49%
        proportions=[0.2038, 0.2844, 0.1517, 0.0892, 0.1349],
        seed=877052)
    data['all_rav4'] = generate_distribution('RAV4_2019_FIT_DENOISED_ACCEL')

    df = pd.DataFrame(data).reset_index().rename(columns={'index': 'rank'})
    df.to_csv('distributions.csv', index=False)
    upload_to_s3(
        'circles.data.pipeline',
        'fact_vehicle_distributions/'
        'distributions.csv',
        'distributions.csv'
    )
    os.remove('distributions.csv')
