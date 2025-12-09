# How to Get Your Hugging Face Token & Access ESM3

## 1. Get Your Token
1. Go to **[Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens)**.
2. Click **"Create new token"**.
3. Name it (e.g., `jinja-esm3`).
4. Select **"Read"** permissions (default is usually fine).
5. Click **"Create token"** and copy the string (starts with `hf_...`).

## 2. Request Model Access
The ESM3 model is gated. You must accept the license agreement:
1. Go to **[evolutionaryscale/esm3-sm-open-v1](https://huggingface.co/evolutionaryscale/esm3-sm-open-v1)**.
2. Log in if needed.
3. Review the license and click **"Agree and Access Repository"**.
   - *Note: It may take a moment to be approved, but often it's instant.*

## 3. Configure Your Environment
Run the following command in your terminal (replace `hf_...` with your actual token):

```bash
export HF_TOKEN=hf_your_token_here
```

To make it permanent, add it to your `~/.bashrc`:
```bash
echo "export HF_TOKEN=hf_your_token_here" >> ~/.bashrc
source ~/.bashrc
```
