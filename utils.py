# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import zipfile

import numpy as np
import pandas as pd


class M5Data:
    """
    A helper class to read M5 source files and create submissions.

    :param str data_path: Path to the folder that contains M5 data files, which is
        either a single `.zip` file or some `.csv` files extracted from that zip file.
    """
    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

    def __init__(self, data_path=None):
        self.data_path = os.path.abspath("data") if data_path is None else data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"There is no folder '{self.data_path}'.")

        acc_path = os.path.join(self.data_path, "m5-forecasting-accuracy.zip")
        unc_path = os.path.join(self.data_path, "m5-forecasting-uncertainty.zip")
        self.acc_zipfile = zipfile.ZipFile(acc_path) if os.path.exists(acc_path) else None
        self.unc_zipfile = zipfile.ZipFile(unc_path) if os.path.exists(unc_path) else None

        self._sales_df = None
        self._calendar_df = None
        self._prices_df = None

    @property
    def num_items(self):
        return self.sales_df.shape[0]

    @property
    def num_aggregations(self):
        return 42840

    @property
    def num_days(self):
        return self.sales_df.shape[1] - 5 + 2 * 28

    @property
    def num_items_by_state(self):
        return self.sales_df["state_id"].value_counts().to_dict()

    @property
    def event_types(self):
        return ["Cultural", "National", "Religious", "Sporting"]

    @property
    def sales_df(self):
        if self._sales_df is None:
            self._sales_df = self._read_csv("sales_train_validation.csv", index_col=0)
        return self._sales_df

    @property
    def calendar_df(self):
        if self._calendar_df is None:
            self._calendar_df = self._read_csv("calendar.csv", index_col=0)
        return self._calendar_df

    @property
    def prices_df(self):
        if self._prices_df is None:
            df = self._read_csv("sell_prices.csv")
            df["id"] = df.item_id + "_" + df.store_id + "_validation"
            df = pd.pivot_table(df, values="sell_price", index="id", columns="wm_yr_wk")
            self._prices_df = df.fillna(0.).loc[self.sales_df.index]
        return self._prices_df

    def listdir(self):
        """
        List all files in `self.data_path` folder.
        """
        files = set(os.listdir(self.data_path))
        if self.acc_zipfile:
            files |= set(self.acc_zipfile.namelist())
        if self.unc_zipfile:
            files |= set(self.unc_zipfile.namelist())
        return files

    def _read_csv(self, filename, index_col=None, use_acc_file=True):
        """
        Returns the dataframe from csv file ``filename``.

        :param str filename: name of the file with trailing `.csv`.
        :param int index_col: indicates which column from csv file is considered as index.
        :param bool acc_file: whether to load data from accuracy.zip file or uncertainty.zip file.
        """
        assert filename.endswith(".csv")
        if filename not in self.listdir():
            raise FileNotFoundError(f"Cannot find either '{filename}' "
                                    "or 'm5-forecasting-accuracy.zip' file "
                                    f"in '{self.data_path}'.")

        if use_acc_file and self.acc_zipfile and filename in self.acc_zipfile.namelist():
            return pd.read_csv(self.acc_zipfile.open(filename), index_col=index_col)

        if self.unc_zipfile and filename in self.unc_zipfile.namelist():
            return pd.read_csv(self.acc_zipfile.open(filename), index_col=index_col)

        return pd.read_csv(os.path.join(self.data_path, filename), index_col=index_col)

    def get_sales(self):
        """
        Returns `sales` np.array with shape `num_items x num_train_days`.
        """
        return self.sales_df.iloc[:, 5:].values

    def get_prices(self):
        """
        Returns `prices` np.array with shape `num_items x num_days`.

        In some days, there are some items not available, so their prices will be NaN.

        :param float fillna: a float value to replace NaN. Defaults to 0.
        """
        x = self.prices_df.values
        x = np.repeat(x,repeats=7,axis=-1)[:, :self.calendar_df.shape[0]]
        assert x.shape == (self.num_items, self.num_days)
        return x

    def get_snap(self):
        """
        Returns a `num_days x 3` boolean tensor which indicates whether
        SNAP purchases are allowed at a state in a particular day. The order
        of the first dimension indicates the states "CA", "TX", "WI" respectively.

        """
        snap = self.calendar_df[["snap_CA", "snap_TX", "snap_WI"]].values
        x = snap
        assert x.shape == (self.num_days, 3)
        return x

    def get_event(self, by_types=False):
        """
        Returns a tensor with length `num_days` indicating whether there are
        special events on a particular day.

        There are 4 types of events: "Cultural", "National", "Religious", "Sporting".

        :param bool by_types: if True, returns a `num_days x 4` tensor indicating
            special event by type. Otherwise, only returns a `num_days x 1` tensor indicating
            whether there is a special event.
        """
        if not by_types:
            event = self.calendar_df["event_type_1"].notnull().values[..., None]
            x = event
            assert x.shape == (self.num_days, 1)
            return x

        types = self.event_types
        event1 = pd.get_dummies(self.calendar_df["event_type_1"])[types].astype(bool)
        event2 = pd.DataFrame(columns=types)
        types2 = ["Cultural", "Religious"]
        event2[types2] = pd.get_dummies(self.calendar_df["event_type_2"])[types2].astype(bool)
        event2.fillna(False, inplace=True)
        x = (event1.values | event2.values)
        assert x.shape == (self.num_days, 4)
        return x

    def get_dummy_day_of_month(self):
        """
        Returns dummy day of month tensor with shape `num_days x 31`.
        """
        dom = pd.get_dummies(pd.to_datetime(self.calendar_df.index).day).values
        x = dom
        assert x.shape == (self.num_days, 31)
        return x

    def get_dummy_month_of_year(self):
        """
        Returns dummy month of year tensor with shape `num_days x 12`.
        """
        moy = pd.get_dummies(pd.to_datetime(self.calendar_df.index).month).values
        x = moy
        assert x.shape == (self.num_days, 12)
        return x

    def get_dummy_day_of_week(self):
        """
        Returns dummy day of week tensor with shape `num_days x 7`.
        """
        dow = pd.get_dummies(self.calendar_df.wday).values
        x = dow
        assert x.shape == (self.num_days, 7)
        return x

    def get_christmas(self):
        """
        Returns a boolean 1D tensor with length `num_days` indicating if that day is
        Chrismas.
        """
        christmas = self.calendar_df.index.str.endswith("12-25")[..., None]
        x = christmas.astype(int)
        assert x.shape == (self.num_days, 1)
        return x

    def get_aggregated_sales(self, state=True, store=True, cat=True, dept=True, item=True):
        """
        Returns aggregated sales at a particular aggregation level.

        The result will be a tensor with shape `num_timeseries x num_train_days`.
        """
        groups = []
        if not state:
            groups.append("state_id")
        if not store:
            groups.append("store_id")
        if not cat:
            groups.append("cat_id")
        if not dept:
            groups.append("dept_id")
        if not item:
            groups.append("item_id")

        if len(groups) > 0:
            x = self.sales_df.groupby(groups, sort=False).sum().values
        else:
            x = self.sales_df.iloc[:, 5:].sum().values[None, :]
        return x

    def get_aggregated_ma_dollar_sales(self, state=True, store=True,
                                       cat=True, dept=True, item=True):
        """
        Returns aggregated "moving average" dollar sales at a particular aggregation level
        during the last 28 days.

        The result can be used as `weight` for evaluation metrics.
        """
        groups = []
        if not state:
            groups.append("state_id")
        if not store:
            groups.append("store_id")
        if not cat:
            groups.append("cat_id")
        if not dept:
            groups.append("dept_id")
        if not item:
            groups.append("item_id")

        prices = self.prices_df.values.repeat(7, axis=1)[:, :self.sales_df.shape[1] - 5]
        df = (self.sales_df.iloc[:, 5:] * prices).T.rolling(28, min_periods=1).mean().T

        if len(groups) > 0:
            for g in groups:
                df[g] = self.sales_df[g]
            x = df.groupby(groups, sort=False).sum().values
        else:
            x = df.sum().values[None, :]
        return x

    def get_all_aggregated_sales(self):
        """
        Returns aggregated sales for all aggregation levels.
        """
        xs = []
        for level in self.aggregation_levels:
            xs.append(self.get_aggregated_sales(**level))
        xs = np.concatenate(xs,axis=0)
        assert xs.shape[0] == self.num_aggregations
        return xs

    def get_all_aggregated_ma_dollar_sales(self):
        """
        Returns aggregated "moving average" dollar sales for all aggregation levels.
        """
        xs = []
        for level in self.aggregation_levels:
            xs.append(self.get_aggregated_ma_dollar_sales(**level))
        xs = np.concatenate(xs,axis=0)
        assert xs.shape[0] == self.num_aggregations
        return xs

    @property
    def aggregation_levels(self):
        """
        Returns the list of all aggregation levels.
        """
        return [
            {"state": True,  "store": True,  "cat": True,  "dept": True,  "item": True},
            {"state": False, "store": True,  "cat": True,  "dept": True,  "item": True},
            {"state": True,  "store": False, "cat": True,  "dept": True,  "item": True},
            {"state": True,  "store": True,  "cat": False, "dept": True,  "item": True},
            {"state": True,  "store": True,  "cat": True,  "dept": False, "item": True},
            {"state": False, "store": True,  "cat": False, "dept": True,  "item": True},
            {"state": False, "store": True,  "cat": True,  "dept": False, "item": True},
            {"state": True,  "store": False, "cat": False, "dept": True,  "item": True},
            {"state": True,  "store": False, "cat": True,  "dept": False, "item": True},
            {"state": True,  "store": True,  "cat": True,  "dept": True,  "item": False},
            {"state": False, "store": True,  "cat": True,  "dept": True,  "item": False},
            {"state": False, "store": False, "cat": False, "dept": False, "item": False},
        ]

    def make_accuracy_submission(self, filename, prediction):
        """
        Makes submission file given prediction result.

        :param str filename: name of the submission file.
        :param np.array predicition: the prediction tensor with shape `num_items x 28`.
        """
        df = self._read_csv("sample_submission.csv", index_col=0)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (self.num_items, 28)
        # the later 28 days only available 1 month before the deadline
        assert df.shape[0] == prediction.shape[0] * 2
        df.iloc[:prediction.shape[0], :] = prediction
        df.to_csv(filename)

    def make_uncertainty_submission(self, filename, prediction):
        """
        Makes submission file given prediction result.

        :param str filename: name of the submission file.
        :param np.array predicition: the prediction tensor with shape
            `9 x num_aggregations x 28`. The first dimension indicates
            9 quantiles defined in `self.quantiles`. The second dimension
            indicates aggreated series defined in `self.aggregation_levels`,
            with corresponding order. This is also the order of
            submission file.
        """
        df = self._read_csv("sample_submission.csv", index_col=0, use_acc_file=False)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (9, self.num_aggregations, 28)
        prediction = prediction.reshape(-1, 28)
        # the later 28 days only available 1 month before the deadline
        assert df.shape[0] == prediction.shape[0] * 2
        df.iloc[:prediction.shape[0], :] = prediction
        df.to_csv(filename)


# class BatchDataLoader:
#     """
#     DataLoader class which iterates over the dataset (data_x, data_y) in batch.
#
#     Usage::
#
#         >>> data_loader = BatchDataLoader(data_x, data_y, batch_size=1000)
#         >>> for batch_x, batch_y in data_loader:
#         ...     # do something with batch_x, batch_y
#     """
#     def __init__(self, data_x, data_y, batch_size, shuffle=True):
#         super().__init__()
#         self.data_x = data_x
#         self.data_y = data_y
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         assert self.data_x.size(0) == self.data_y.size(0)
#         assert len(self) > 0
#
#     @property
#     def size(self):
#         return self.data_x.size(0)
#
#     def __len__(self):
#         # XXX: should we remove or include the tailing data (which has len < batch_size)?
#         return math.ceil(self.size / self.batch_size)
#
#     def _sample_batch_indices(self):
#         if self.shuffle:
#             idx = torch.randperm(self.size)
#         else:
#             idx = torch.arange(self.size)
#         return idx, len(self)
#
#     def __iter__(self):
#         idx, n_batches = self._sample_batch_indices()
#         for i in range(n_batches):
#             _slice = idx[i * self.batch_size: (i + 1) * self.batch_size]
#             yield self.data_x[_slice], self.data_y[_slice]
