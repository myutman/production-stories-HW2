import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering

import sqlite3


def read_data(file):
    con = sqlite3.connect(file)
    cur = con.cursor()
    cur.execute("""
    SELECT
        b.date, a.time, a.lot_size, a.price, b.id, trading_type
    FROM trading_session AS b INNER JOIN chart_data AS a
    ON a.session_id == b.id ORDER BY b.id, date, time""")
    df = pd.DataFrame()
    datetimes = []
    sizes = []
    prices = []
    session_ids = []
    ttypes = []

    for record in cur.fetchall():
        date, time, size, price, id_, ttype = record
        datetimes.append(datetime.datetime.fromisoformat(date + " " + time))
        sizes.append(size)
        prices.append(price)
        session_ids.append(id_)
        ttypes.append(ttype)

    df["datetime"] = datetimes
    df["size"] = sizes
    df["price"] = prices
    df["session_id"] = session_ids
    df["ttype"] = ttypes

    return df[df["ttype"] == "daily"], df[df["ttype"] == "monthly"]


def group_by_sesion_id(df):
    groups = []
    for key, group in df.groupby(by=["session_id"]):
        groups.append((key, group))
    return groups


def vectorize_groups(groups):
    def get_key(group):
        dt = datetime.datetime.fromisoformat(str(group["datetime"].min()))
        return dt.date()#, dt.hour

    groups = list({get_key(x[1]): x for x in groups}.values())
    latest_price = np.array(groups[0][1]["price"])[-1]

    vecs = []
    for key, group_ in groups[1:]:
        group = group_.copy()
        avg_price = (group["price"] * group["size"]).sum() / group["size"].sum()
        group["price"] -= avg_price
        dt = datetime.datetime.fromisoformat(str(group["datetime"].min()))
        dt = dt.replace(minute=0, second=0)
        vec = []
        for i in range(60):
            subgroup = group[group["datetime"] < dt + datetime.timedelta(minutes=i + 1)]
            latest_price = latest_price if len(subgroup) == 0 else np.array(subgroup["price"])[-1]
            vec.append(latest_price)
        vec = np.array(vec)
        vecs.append((key, vec))
    return vecs


def build_clusters(vecs, dist, norm, n_clusters=None, distance_threshold=None):
    model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, affinity="precomputed", linkage="complete")

    
    distances = [[dist(norm(v1), norm(v2)) for _, v2 in vecs] for _, v1 in vecs]
            
    model.fit(distances)
    return model.labels_

def show_clusters(groups, clusters):
    def add_cluster(x, cluster_id):
        x["cluster_id"] = cluster_id
        return x

    colors = [
        'b',
        'g',
        'r',
        'c',
        'm',
        'y',
        'k',
        'orange'
    ]

    fig = plt.figure(figsize=(20, 4))

    for cluster_id, (_, x) in zip(clusters, groups):
        total_df = add_cluster(x, cluster_id)
        plt.plot(total_df.index, total_df["price"], color=colors[cluster_id])
    
    return fig


def show_clusters_dynamics(vecs, groups, dist, norm):
    clusters = build_clusters(
        vecs=vecs,
        dist=dist, 
        norm=norm,
        n_clusters=5
    )
    fig1 = show_clusters(groups, clusters)

    clusters = build_clusters(
        vecs=vecs,
        dist=dist, 
        norm=norm,
        n_clusters=6
    )
    fig2 = show_clusters(groups, clusters)

    clusters = build_clusters(
        vecs=vecs,
        dist=dist, 
        norm=norm,
        n_clusters=7
    )
    fig3 = show_clusters(groups, clusters)