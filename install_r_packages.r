repos <- "https://cloud.r-project.org"
needs <- c("iNEXT")   # add "iNEXT.4beta" if you want
to_install <- setdiff(needs, rownames(installed.packages()))
if (length(to_install)) {
  install.packages(to_install, repos = repos)
}
# sanity print
for (pkg in needs) {
  cat(pkg, "version:", as.character(packageVersion(pkg)), "\n")
}
