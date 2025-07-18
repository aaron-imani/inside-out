import numpy as np
import plotly.graph_objects as go
import torch
from classifier_manager import ClassifierManager
from model_generation import ModelGeneration
from plotly.subplots import make_subplots
from reduction import Reduction
from torch.nn.functional import softmax


def create_2dlist(n_row: int, n_col: int) -> list[list]:
    return [[[] for _ in range(n_col)] for _ in range(n_row)]


def create_layer_needed(n_layer: int, step: int) -> list[int]:
    ret = [n_layer - 1 - i for i in range(0, n_layer, step)]
    ret.reverse()
    return ret


def plot_difference(
    hs_before: list[torch.Tensor],
    hs_after: list[torch.Tensor],
    clfr: ClassifierManager,
    llm: ModelGeneration,
    token_range: int = 20,
    step: int = 4,
    topk: int = 3,
):
    infos = build_difference_data(
        hs_before, hs_after, clfr, llm, token_range, step, topk
    )

    hovertext_with_gaps = []
    for before, after in zip(infos["info_before"], infos["info_after"]):
        hovertext_with_gaps.extend([before, after, [""] * len(before)])
    if hovertext_with_gaps and not hovertext_with_gaps[-1]:
        hovertext_with_gaps.pop()

    P_m = infos["P_m"]
    Layers = infos["row_labels"]

    P_m_with_gaps = []
    Layers_with_gaps = []

    for i, row in enumerate(P_m):
        P_m_with_gaps.append(row)
        Layers_with_gaps.append(Layers[i])

        if i % 2 == 1 and i < len(P_m) - 1:
            P_m_with_gaps.append([None] * len(row))
            Layers_with_gaps.append("\u200b" * (i + 1))

    text_with_gaps = [
        [f"{value * 100:.1f}%" if value is not None else " " for value in row]
        for row in P_m_with_gaps
    ]

    colorscale = [[0.0, "green"], [0.5, "white"], [1.0, "red"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=P_m_with_gaps,
            y=Layers_with_gaps,
            colorscale=colorscale,
            customdata=hovertext_with_gaps,
            hovertemplate="%{customdata}<extra></extra>",
            text=text_with_gaps,
            texttemplate="%{text}",
            colorbar=dict(title="P_m", titleside="right"),
        )
    )

    cell_size = 40
    width = cell_size * token_range * 1.5 + 300
    height = cell_size * len(Layers_with_gaps) * 0.6 + 200

    fig.update_layout(
        width=width,
        height=height,
        xaxis=dict(
            tickvals=list(range(len(infos["col_labels"]))),
            ticktext=infos["col_labels"],
            constrain="domain",
        ),
        yaxis=dict(
            autorange="reversed",
            ticktext=Layers_with_gaps,
        ),
        title=dict(
            text="P_m of Each Token in Each Layer Before and After Perturbation",
            x=0.5,
            xanchor="center",
        ),
    )

    fig.update_traces(
        hoverinfo="skip",
        zsmooth=False,
        showscale=True,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        connectgaps=False,
        colorbar=dict(
            title=" ",
            tickvals=[0, 1],
            ticktext=["Safe", "Malicious"],
            len=(
                0.8
                if step > llm.llm_cfg.n_layer - 2
                else -0.9 * np.exp(-0.2 * step) + 1
            ),
        ),
    )

    fig.show()


def build_difference_data(
    hs_before: list[torch.Tensor],
    hs_after: list[torch.Tensor],
    clfr: ClassifierManager,
    llm: ModelGeneration,
    token_range: int = 20,
    step: int = 4,
    topk: int = 3,
):
    hs_before = hs_before[:token_range]
    hs_after = hs_after[:token_range]

    layer_needed = create_layer_needed(llm.llm_cfg.n_layer, step)
    n_layer_needed = len(layer_needed)
    n_row = 2 * len(layer_needed)
    n_col = len(hs_before)

    unembedded = lambda x: llm.model.lm_head(llm.model.model.norm(x))
    decoder = llm.tokenizer.decode
    greedy_decoder = lambda x: decoder(torch.argmax(x))

    row_labels = [
        f"Layer {layer} {'Before' if i % 2 == 0 else 'After'}"
        for layer in layer_needed
        for i in range(2)
    ]
    col_labels = [
        greedy_decoder(unembedded(hs_before[i][-1])) for i in range(len(hs_before))
    ]

    P_m = create_2dlist(n_row, n_col)
    distribution_before = create_2dlist(n_layer_needed, n_col)
    distribution_after = create_2dlist(n_layer_needed, n_col)
    info_before = create_2dlist(n_layer_needed, n_col)
    info_after = create_2dlist(n_layer_needed, n_col)

    for i, layer in enumerate(layer_needed):
        for token in range(n_col):
            embedding_before = hs_before[token][layer, :]
            embedding_after = hs_after[token][layer, :]

            P_m[2 * i][token] = (
                int(
                    clfr.classifiers[layer]
                    .predict_proba(embedding_before.unsqueeze(0))
                    .item()
                    * 1000
                )
                / 1000
            )
            P_m[2 * i + 1][token] = (
                int(
                    clfr.classifiers[layer]
                    .predict_proba(embedding_after.unsqueeze(0))
                    .item()
                    * 1000
                )
                / 1000
            )

            distribution_before[i][token] = (
                softmax(unembedded(embedding_before), dim=0) * 100
            )
            distribution_after[i][token] = (
                softmax(unembedded(embedding_after), dim=0) * 100
            )

            difference = distribution_after[i][token] - distribution_before[i][token]

            topk_before = torch.topk(distribution_before[i][token], topk)
            topk_after = torch.topk(distribution_after[i][token], topk)

            topk_difference_upper = torch.topk(difference, topk)
            topk_difference_lower = torch.topk(-difference, topk)

            info_before[i][token] = f"<b>Top-{topk} logprobs</b><br><br>" + "<br>".join(
                [
                    f"{decoder([j.item()])}: {p.item():.2f}%"
                    for j, p in zip(topk_before.indices, topk_before.values)
                ]
            )

            info_after[i][token] = (
                f"<b>Top-{topk} logprobs</b><br><br>"
                + "<br>".join(
                    [
                        f"{decoder([j.item()])}: {p.item():.2f}%"
                        for j, p in zip(topk_after.indices, topk_after.values)
                    ]
                )
                + f"<br><br><b>Top-{topk} logprobs increasing</b><br><br>"
                + "<br>".join(
                    [
                        f"{decoder([j.item()])}: {p.item():.2f}%"
                        for j, p in zip(
                            topk_difference_upper.indices, topk_difference_upper.values
                        )
                    ]
                )
                + f"<br><br><b>Top-{topk} logprobs decreasing</b><br><br>"
                + "<br>".join(
                    [
                        f"{decoder([j.item()])}: {p.item():.2f}%"
                        for j, p in zip(
                            topk_difference_lower.indices, topk_difference_lower.values
                        )
                    ]
                )
            )

    return {
        "P_m": P_m,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "info_before": info_before,
        "info_after": info_after,
    }


def plot_testacc(testacc: list[float], threshold: float):
    fig = go.Figure()
    testacc = [i * 100 for i in testacc]
    threshold = threshold * 100

    fig.add_trace(
        go.Scatter(
            x=list(range(len(testacc))),
            y=testacc,
            mode="lines+markers",
            name="Test Accuracy (%)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, len(testacc) - 1],
            y=[threshold, threshold],
            mode="lines",
            name="Threshold",
        )
    )

    fig.update_layout(
        title=dict(text="CAV Test Accuracy of Each Layer", x=0.5, xanchor="center"),
        xaxis_title="Layer",
        yaxis_title="Test Accuracy (%)",
        width=600,
        height=400,
    )

    fig.show()


def plot_reduction(
    insts,
    pos_train_embds,
    neg_train_embds,
    pos_test_embds,
    neg_test_embds,
    n_layer,
    char_range=30,
):
    c = 8
    r = n_layer // c + 1 if n_layer % c else n_layer // c

    fig = make_subplots(
        rows=r, cols=c, subplot_titles=[f"Layer {i+1}" for i in range(n_layer)]
    )

    for i in range(r):
        for j in range(c):
            layer = i * c + j
            if layer >= n_layer:
                break
            pca = Reduction(120)

            # train_data = torch.vstack(
            #     [pos_train_embds.layers[layer], neg_train_embds.layers[layer]]
            # )
            all_data = torch.vstack(
                [
                    pos_train_embds.layers[layer],
                    neg_train_embds.layers[layer],
                    pos_test_embds.layers[layer],
                    neg_test_embds.layers[layer],
                ]
            )
            pca.fit(all_data)

            # pos_train_pca = pca.transform(pos_train_embds.layers[layer])
            # neg_train_pca = pca.transform(neg_train_embds.layers[layer])
            # pos_test_pca = pca.transform(pos_test_embds.layers[layer])
            # neg_test_pca = pca.transform(neg_test_embds.layers[layer])

            all_pos = torch.vstack(
                [pos_train_embds.layers[layer], pos_test_embds.layers[layer]]
            )
            all_neg = torch.vstack(
                [neg_train_embds.layers[layer], neg_test_embds.layers[layer]]
            )
            # Transform all data to PCA space
            all_pos = pca.transform(all_pos)
            all_neg = pca.transform(all_neg)

            pos_tsne = pca.get_tsne(all_pos)
            neg_tsne = pca.get_tsne(all_neg)
            pos_x, pos_y, neg_x, neg_y = (
                pos_tsne[:, 0],
                pos_tsne[:, 1],
                neg_tsne[:, 0],
                neg_tsne[:, 1],
            )

            # pos_x, pos_y, neg_x, neg_y = (
            #     all_pos[:, 0],
            #     all_pos[:, 1],
            #     all_neg[:, 0],
            #     all_neg[:, 1],
            # )

            # Stack train data and create labels
            # pos = pos_train_embds.layers[layer]
            # neg = neg_train_embds.layers[layer]
            # X_train = torch.vstack([pos, neg])
            # # y_train = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])])

            # pca.fit(X_train)
            # X_train_pca = pca.transform(X_train)

            # # Project classifier's weights to PCA space
            # weight, bias = (
            #     clf.classifiers[layer].linear.coef_,
            #     clf.classifiers[layer].linear.intercept_,
            # )
            # w_pca = pca.transform(weight)[0]

            # # Compute decision boundary: w_pca[0]*x + w_pca[1]*y + bias = 0
            # x_vals = torch.linspace(
            #     X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 100
            # )
            # x_vals_np = (
            #     x_vals.detach().cpu().numpy()
            #     if hasattr(x_vals, "detach")
            #     else np.array(x_vals)
            # )
            # w_pca_np = (
            #     w_pca.detach().cpu().numpy()
            #     if hasattr(w_pca, "detach")
            #     else np.array(w_pca)
            # )
            # bias_np = (
            #     bias.detach().cpu().numpy()
            #     if hasattr(bias, "detach")
            #     else np.array(bias)
            # )
            # y_vals = -(w_pca_np[0] * x_vals_np + bias_np) / (
            #     w_pca_np[1] + 1e-10
            # )  # Avoid division by 0

            # # Plot decision boundary
            # fig.add_trace(
            #     go.Scatter(
            #         x=x_vals.tolist(),
            #         y=y_vals.tolist(),
            #         mode="lines",
            #         line=dict(color="black", dash="dash"),
            #         name="Decision Boundary",
            #         legendgroup="Decision Boundary",
            #         showlegend=(i == 0 and j == 0),
            #     ),
            #     row=i + 1,
            #     col=j + 1,
            # )
            insts["train"][0] = [inst[:char_range] for inst in insts["train"][0]]
            insts["train"][1] = [inst[:char_range] for inst in insts["train"][1]]
            insts["test"][0] = [inst[:char_range] for inst in insts["test"][0]]
            insts["test"][1] = [inst[:char_range] for inst in insts["test"][1]]

            row, col = i + 1, j + 1

            fig.add_trace(
                go.Scatter(
                    x=pos_x,
                    y=pos_y,
                    mode="markers",
                    marker=dict(color="green", opacity=0.3),
                    name="Positive",
                    legendgroup="Positive",
                    text=insts["train"][0] + insts["test"][0],
                    hoverinfo="text",
                    showlegend=(i == 0 and j == 0),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=neg_x,
                    y=neg_y,
                    mode="markers",
                    marker=dict(color="red", opacity=0.3),
                    name="Negative",
                    legendgroup="Negative",
                    text=insts["train"][1] + insts["test"][1],
                    hoverinfo="text",
                    showlegend=(i == 0 and j == 0),
                ),
                row=row,
                col=col,
            )
            # fig.add_trace(
            #     go.Scatter(
            #         x=pos_test_pca[:, 0],
            #         y=pos_test_pca[:, 1],
            #         mode="markers",
            #         marker=dict(color="orange", opacity=0.3),
            #         name="Positive Test",
            #         legendgroup="Positive Test",
            #         text=insts["test"][0],
            #         hoverinfo="text",
            #         showlegend=(i == 0 and j == 0),
            #     ),
            #     row=row,
            #     col=col,
            # )
            # fig.add_trace(
            #     go.Scatter(
            #         x=neg_test_pca[:, 0],
            #         y=neg_test_pca[:, 1],
            #         mode="markers",
            #         marker=dict(color="green", opacity=0.3),
            #         name="Negative Test",
            #         legendgroup="Negative Test",
            #         text=insts["test"][1],
            #         hoverinfo="text",
            #         showlegend=(i == 0 and j == 0),
            #     ),
            #     row=row,
            #     col=col,
            # )

    fig.update_layout(
        title=dict(text="t-SNE of Embeddings Across Layers", x=0.5, xanchor="center"),
        height=r * 200,
        width=c * 200,
        showlegend=True,
    )

    # fig.show()
    return fig
