# -*- coding: utf-8 -*-
"""
中国场外公募基金（场外基金）净值拉取与绩效分析脚本
====================================================
用途：
    1. 批量拉取国内场外基金历史净值（天天财富/东方财富公开数据）
    2. 计算常用绩效指标：累计收益率、年化收益、最大回撤、波动率、夏普比率
    3. 绘制净值走势图并保存为图片
    4. 导出结果为 CSV，便于后续做行为金融 / 投资者行为研究

依赖安装（命令行执行一次即可）：
    pip install akshare pandas matplotlib

数据说明：
    本脚本数据来自公开渠道（东方财富天天基金），仅用于学术研究与投资者教育，
    不构成任何投资建议。原始数据不入库，运行时实时拉取。
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# ---------- 中文显示配置（防止画图乱码）----------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ==================================================
# 1. 拉取单只基金的历史净值
# ==================================================
def fetch_fund_nav(fund_code: str, period: str = "累计净值") -> pd.DataFrame:
    """
    拉取单只场外基金历史净值。

    参数:
        fund_code: 基金代码，如 "110011"（易方达优质精选）
        period:    "单位净值" 或 "累计净值"（累计净值更能反映真实长期表现）

    返回:
        DataFrame，含 [日期, 净值] 两列，按日期升序
    """
    # akshare 接口：fund_open_fund_info_em(基金代码, 数据类型)
    # 数据类型可选："单位净值走势" / "累计净值走势" / "累计收益率走势"
    nav_type = "累计净值走势" if period == "累计净值" else "单位净值走势"
    df = ak.fund_open_fund_info_em(symbol=fund_code, indicator=nav_type)

    # 统一列名与日期格式（不同接口返回列名可能不同，做一次兼容）
    df.columns = [str(c).strip() for c in df.columns]
    date_col = [c for c in df.columns if "日期" in c][0]
    value_col = [c for c in df.columns if "净值" in c or "收益" in c][0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "nav"]
    out["date"] = pd.to_datetime(out["date"])
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
    return out


# ==================================================
# 2. 计算绩效指标
# ==================================================
def calc_performance(nav: pd.Series, rf_annual: float = 0.015) -> dict:
    """
    给定净值序列，计算常用绩效指标。

    参数:
        nav:        净值序列（按日期升序）
        rf_annual: 无风险年化收益率（默认1.5%）

    返回:
        dict，包含各指标
    """
    nav = nav.dropna()
    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)

    if n_days < 2:
        return {"年化收益": np.nan, "最大回撤": np.nan, "年化波动": np.nan, "夏普比率": np.nan}

    # 累计 / 年化收益
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    years = n_days / 252
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan

    # 最大回撤
    cummax = nav.cummax()
    drawdown = nav / cummax - 1
    max_dd = drawdown.min()

    # 年化波动率 & 夏普
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol and ann_vol != 0 else np.nan

    return {
        "累计收益率": round(total_ret * 100, 2),       # 单位：%
        "年化收益率": round(ann_ret * 100, 2),          # 单位：%
        "最大回撤": round(max_dd * 100, 2),            # 单位：%
        "年化波动率": round(ann_vol * 100, 2),         # 单位：%
        "夏普比率": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "样本天数": n_days,
    }


# ==================================================
# 3. 批量分析 + 画图 + 导出
# ==================================================
def analyze_funds(fund_dict: dict, save_csv: str = "fund_performance.csv"):
    """
    批量分析多只基金。

    参数:
        fund_dict: {基金代码: 基金名称}，例如 {"110011": "易方达优质精选"}
        save_csv:  绩效结果导出文件名
    """
    nav_panel = {}      # 存放各基金净值序列用于画图
    perf_rows = []      # 存放各基金绩效指标用于汇总表

    for code, name in fund_dict.items():
        print(f"正在拉取 {code} {name} ...")
        try:
            df = fetch_fund_nav(code)
            nav_panel[code] = df.set_index("date")["nav"].rename(name)
            perf = calc_performance(df["nav"])
            perf["基金代码"] = code
            perf["基金名称"] = name
            perf_rows.append(perf)
            print(f"  -> 拉取成功，共 {len(df)} 个交易日，最新净值 {df['nav'].iloc[-1]:.4f}")
        except Exception as e:
            print(f"  -> 拉取失败：{e}")

    # 汇总表
    perf_df = pd.DataFrame(perf_rows)
    perf_df = perf_df[["基金代码", "基金名称", "累计收益率", "年化收益率",
                       "最大回撤", "年化波动率", "夏普比率", "样本天数"]]
    perf_df.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"\n绩效汇总已导出：{save_csv}")
    print(perf_df.to_string(index=False))

    # 归一化净值曲线（起点=1，便于多基金对比）
    if nav_panel:
        nav_df = pd.concat(nav_panel.values(), axis=1)
        norm_nav = nav_df / nav_df.iloc[0]
        plt.figure(figsize=(10, 6))
        for col in norm_nav.columns:
            plt.plot(norm_nav.index, norm_nav[col], label=col, linewidth=1.5)
        plt.title("场外基金累计净值走势（起点归一=1）")
        plt.xlabel("日期")
        plt.ylabel("归一化净值")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("fund_nav_curve.png", dpi=150)
        print("净值走势图已保存：fund_nav_curve.png")


# ==================================================
# 主程序：在这里改你要研究的基金
# ==================================================
if __name__ == "__main__":
    # 👇 改成你想研究的基金代码与名称（场外基金在天天基金APP可查代码）
    MY_FUNDS = {
        "110011": "易方达优质精选混合",
        "161725": "招商中证白酒指数",
        "005827": "易方达蓝筹精选混合",
        "260108": "景顺长城新兴成长混合",
    }

    analyze_funds(MY_FUNDS)
#（注：内容由AI生成）
