# Troubleshooting

If you’re having issues with **Quantreo**, here are the most common fixes.

---

## Installation problems

**`ModuleNotFoundError: No module named 'quantreo'`**  
→ Make sure the library is installed and your virtual env is active:

```bash
pip install quantreo
```

---
**`ImportError: cannot import name ...`**  
→ You’re using an old version. Update Quantreo:

```bash
pip install -U quantreo
```

---

## Compatibility

Quantreo officially supports **Python 3.10 – 3.13**. Use 3.11 for full stability.

---

## Runtime errors

**`KeyError: 'close'`**  
→ Your DataFrame must have the correct column names or specify them explicitly:

```python
future_returns_sign(df, close_col="Close")
```


---

## Still stuck?

- Open an [Issue on GitHub](https://github.com/Quantreo/quantreo/issues)  
- Or contact us directly → **lucas@quantreo.com**

---

> **Tip:** Always use the latest version  
> ```bash
> pip install -U quantreo
> ```
