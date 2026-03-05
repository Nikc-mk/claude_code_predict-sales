"""
model.py — FT-Transformer (Feature Tokenizer Transformer) для прогнозирования продаж.

Архитектура (Feature Tokenizer + Transformer Encoder):
  - Каждый числовой признак xi → feature-specific Linear(1 → d_model) = token_i
  - Категориальный признак    → Embedding(n_cats, d_model)              = token_cat
  - [CLS] токен               → обучаемый параметр                     = token_cls
  - Stack: (B, 19, d_model)  [17 числовых + 1 cat + 1 CLS]
  - TransformerEncoder: attention между признаками (не между временными шагами!)
  - Выход CLS-токена → две независимые головы → 2 числа (predicted_remaining_sales, predicted_remaining_profit)

Attention здесь моделирует взаимодействие признаков:
  "Если days_left мало, а cumulative_sales высокое → остаток пропорционален недельному тренду".
"""

import torch
import torch.nn as nn

from config import MODEL_CONFIG, NUM_NUMERICAL_FEATURES


class FTTransformerModel(nn.Module):
    """
    Feature Tokenizer Transformer для регрессии на табличных данных.

    Параметры
    ---------
    n_categories : int
        Число уникальных категорий товаров.
    n_num_features : int
        Число числовых признаков (из config.NUM_NUMERICAL_FEATURES).
    d_model : int
        Размерность токена/эмбеддинга.
    nhead : int
        Число голов в MultiHeadAttention.
    num_layers : int
        Число слоёв TransformerEncoder.
    dim_feedforward : int
        Размер скрытого слоя FFN внутри TransformerEncoder.
    dropout : float
    """

    def __init__(
        self,
        n_categories: int,
        n_num_features: int = NUM_NUMERICAL_FEATURES,
        d_model: int = MODEL_CONFIG["d_model"],
        nhead: int = MODEL_CONFIG["nhead"],
        num_layers: int = MODEL_CONFIG["num_layers"],
        dim_feedforward: int = MODEL_CONFIG["dim_feedforward"],
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super().__init__()
        self.d_model = d_model
        self.n_num_features = n_num_features

        # --- Feature Tokenizer для числовых признаков ---
        # Каждый признак имеет свои веса W_i: Linear(1 → d_model) без bias (bias отдельный)
        self.num_tokenizers = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_num_features)]
        )

        # --- Токенайзер для категории ---
        self.cat_embedding = nn.Embedding(n_categories, d_model)

        # --- [CLS] токен (обучаемый вектор) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- LayerNorm после токенизации ---
        self.token_norm = nn.LayerNorm(d_model)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (B, seq_len, d_model)
            norm_first=True,     # Pre-LN для стабильности обучения
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Дополнительная head для обработки категории ---
        self.cat_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # --- Финальная комбинация ---
        self.fusion = nn.Linear(d_model * 2, d_model)

        # --- Регрессионные головы (отдельные для продаж и прибыли) ---
        self.head_revenue = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.head_profit = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        num_features: torch.Tensor,   # (B, n_num_features)
        category_ids: torch.Tensor,   # (B,)
    ) -> torch.Tensor:                # (B, 2) — [predicted_remaining_sales, predicted_remaining_profit]
        """
        Прямой проход.

        Returns
        -------
        torch.Tensor : shape (B, 2) — [:, 0] = remaining_sales, [:, 1] = remaining_profit
        """
        B = num_features.size(0)

        # --- Токенизация числовых признаков ---
        # Каждый xi: (B, 1) → (B, d_model) → unsqueeze → (B, 1, d_model)
        num_tokens = []
        for i, tokenizer in enumerate(self.num_tokenizers):
            xi = num_features[:, i].unsqueeze(1)   # (B, 1)
            token = tokenizer(xi).unsqueeze(1)      # (B, 1, d_model)
            num_tokens.append(token)

        # Stack числовых токенов: (B, n_num_features, d_model)
        num_tokens = torch.cat(num_tokens, dim=1)

        # --- Токенизация категории ---
        cat_token = self.cat_embedding(category_ids).unsqueeze(1)  # (B, 1, d_model)

        # --- [CLS] токен ---
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)

        # --- Конкатенация всех токенов: [CLS, feat_1, ..., feat_10, cat] ---
        tokens = torch.cat([cls_tokens, num_tokens, cat_token], dim=1)  # (B, 12, d_model)
        tokens = self.token_norm(tokens)

        # --- Transformer Encoder (attention между признаками) ---
        encoded = self.transformer_encoder(tokens)  # (B, 12, d_model)

        # --- Выход CLS-токена ---
        cls_output = encoded[:, 0, :]  # (B, d_model)

        # --- Категориальный выход (берем последний токен - категорию) ---
        cat_output = encoded[:, -1, :]  # (B, d_model)
        cat_output = self.cat_head(cat_output)  # (B, d_model)

        # --- Комбинируем CLS и категорию ---
        combined = torch.cat([cls_output, cat_output], dim=-1)  # (B, d_model * 2)
        combined = self.fusion(combined)  # (B, d_model)

        # --- Регрессионные головы ---
        revenue = self.head_revenue(combined).squeeze(1)  # (B,)
        profit  = self.head_profit(combined).squeeze(1)   # (B,)
        return torch.stack([revenue, profit], dim=1)       # (B, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def build_model(n_categories: int, config: dict = MODEL_CONFIG) -> FTTransformerModel:
    """Фабричная функция для создания модели из конфига."""
    return FTTransformerModel(
        n_categories=n_categories,
        n_num_features=NUM_NUMERICAL_FEATURES,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    )


def save_model(model: FTTransformerModel, path: str, extra: dict | None = None) -> None:
    """
    Сохраняет модель + метаданные в .pt файл.

    extra: словарь с дополнительными данными (n_categories, feature_names, etc.)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_categories": model.cat_embedding.num_embeddings,
            "n_num_features": model.n_num_features,
            "d_model": model.d_model,
            "nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.transformer_encoder.layers),
            "dim_feedforward": model.transformer_encoder.layers[0].linear1.out_features,
            "dropout": model.transformer_encoder.layers[0].dropout.p,
        },
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_model(path: str, device: str = "cpu") -> tuple[FTTransformerModel, dict]:
    """
    Загружает модель из .pt файла.

    Returns
    -------
    model : FTTransformerModel
    checkpoint : dict — весь сохранённый словарь (включая extra-данные)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    model = FTTransformerModel(
        n_categories=cfg["n_categories"],
        n_num_features=cfg["n_num_features"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
