# GitHub Upload Guide

Follow these steps to upload your XAI pipeline to GitHub.

## 📋 Prerequisites

- Git installed on your system ([Download Git](https://git-scm.com/downloads))
- GitHub account ([Sign up](https://github.com/join))

## 🚀 Step-by-Step Instructions

### Step 1: Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 2: Create GitHub Repository

1. Go to [github.com](https://github.com) and log in
2. Click the **+** button (top right) → **New repository**
3. Fill in:
   - **Repository name**: `xai-saas-access-control` (or your preferred name)
   - **Description**: "XAI Interpretability Pipeline for SaaS Access Control"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **Create repository**

### Step 3: Add Files to Git

Open terminal/command prompt in your project directory:

```bash
cd "c:\Users\Anuradha\Downloads\SAAS XAI"

# Check status
git status

# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status
```

**Expected Output:**
```
Changes to be committed:
  new file:   .gitignore
  new file:   LICENSE
  new file:   MODEL_SETUP.md
  new file:   QUICKSTART.md
  new file:   README.md
  new file:   README_XAI.md
  new file:   XAI.py
  new file:   XAI_enhanced.py
  new file:   examples/example_c1_explanation.py
  new file:   examples/example_validation.py
  new file:   llm_translator.py
  new file:   requirements_xai.txt
  new file:   test_xai_install.py
  new file:   xai_explainer.py
  new file:   xai_validator.py
```

> ✅ Large files (*.h5, *.joblib, *.json) should be **excluded** by .gitignore

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: XAI Interpretability Pipeline

- Core XAI explainers (SHAP, LIME, Attention, Reconstruction, Embedding)
- LLM translation layer with stakeholder-specific prompts
- Validation framework (Fidelity & Stability metrics)
- Enhanced wrapper for existing system
- Comprehensive documentation and examples"
```

### Step 5: Link to GitHub Repository

Replace `YOUR_USERNAME` and `REPO_NAME` with your actual values:

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Verify remote
git remote -v
```

**Example:**
```bash
git remote add origin https://github.com/anuradha123/xai-saas-access-control.git
```

### Step 6: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

**If prompted for credentials:**
- Username: Your GitHub username
- Password: Use a [Personal Access Token](https://github.com/settings/tokens) (not your GitHub password)

### Step 7: Verify Upload

1. Go to your GitHub repository URL: `https://github.com/YOUR_USERNAME/REPO_NAME`
2. You should see all files uploaded
3. README.md will be displayed on the main page

## 🎉 Success!

Your XAI pipeline is now on GitHub! 

### Next Steps:

**1. Update README.md with your info:**
```bash
# Edit README.md to replace placeholders:
# - YOUR_USERNAME → your actual GitHub username
# - your.email@example.com → your email
# - [Your Name] in LICENSE → your name

git add README.md LICENSE
git commit -m "Update author information"
git push
```

**2. Add Topics/Tags** (on GitHub website):
- Go to your repository
- Click "⚙️ Settings" (if you own it) or edit description
- Add topics: `explainable-ai`, `xai`, `shap`, `lime`, `security`, `aws`, `cloudtrail`, `bert`, `machine-learning`

**3. Create Release** (optional):
- Go to your repository → Releases → "Create a new release"
- Tag: `v1.0.0`
- Title: "XAI Pipeline v1.0 - Initial Release"
- Description: Summarize features

## 📦 Handling Large Model Files

Large model files (.h5, .joblib, etc.) are **excluded by .gitignore**.

### Option 1: Git LFS (Large File Storage)

If you want to include models:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.joblib"
git lfs track "*.json"

# Add .gitattributes
git add .gitattributes

# Add models
git add autoencoder.h5 isolation_forest.joblib
git commit -m "Add models via Git LFS"
git push
```

> ⚠️ **Warning**: Git LFS has storage limits. Free plan: 1 GB storage, 1 GB bandwidth/month.

### Option 2: External Storage (Recommended)

Host models separately and provide download links in README:

```markdown
## Download Models

Download pre-trained models:
- [autoencoder.h5](https://your-storage-link.com/autoencoder.h5)
- [isolation_forest.joblib](https://your-storage-link.com/isolation_forest.joblib)
- [CloudTrail data](https://your-storage-link.com/cloudtrail.json)

Place in project root directory.
```

Options for hosting:
- Google Drive
- Dropbox
- AWS S3
- Hugging Face Hub (for transformers)
- Kaggle Datasets

## 🔄 Making Changes Later

```bash
# Make your changes to files

# Check what changed
git status
git diff

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add feature X"

# Push to GitHub
git push
```

## 🐛 Troubleshooting

**"Repository not found"**
- Check remote URL: `git remote -v`
- Verify repository exists on GitHub
- Check permissions (private repos)

**"Authentication failed"**
- Use Personal Access Token instead of password
- Generate at: https://github.com/settings/tokens
- Select scopes: `repo` (full control of private repositories)

**"File too large"**
- Check .gitignore is working: `git check-ignore -v <filename>`
- Files >100 MB require Git LFS
- Consider hosting large files externally

**"Cannot push to main"**
- Branch may be protected
- Push to a new branch: `git push -u origin feature-branch`

## 📚 Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git LFS](https://git-lfs.github.com/)
- [Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

---

Need help? Check the [GitHub Community](https://github.community/) or open an issue in your repository.
