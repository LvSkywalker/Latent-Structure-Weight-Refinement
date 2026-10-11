# Uploading the release archive to GitHub

1. Extract the ZIP. The repository files are inside `subspace-weight-refinement/`.
2. Create a new GitHub repository. Suggested name: `latent-structure-weight-refinement`.
   The archive already contains a README, license and ignore rules, so leave
   GitHub's corresponding initialization options unchecked.
3. Use **Add file → Upload files** and upload the contents of the extracted
   folder. `README.md`, `pyproject.toml` and `src/` belong at the repository
   root, rather than inside a second nested project folder. Include the hidden
   `.github/` directory and `.gitignore` file.
4. Commit the upload. The numerical-check workflow can then run on GitHub;
   the release has already passed the corresponding checks locally.

Suggested repository description:

> Exploring latent structure for signed Monte Carlo weight refinement: a reproducible Python study of learned subspaces, kernels and neural networks.

Upload the extracted files, not the ZIP as a single repository file.
