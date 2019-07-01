Notes from Etash:
1. If you get node version problems, use https://www.npmjs.com/package/electron-rebuild
2. If you get get a problem with the java bindings, 
  A. cd into node_modules/java
  B. run node postInstall.js
3. If you have files that are too large to commit, use https://github.com/IBM/BluePic/wiki/Using-BFG-Repo-Cleaner-tool-to-remove-sensitive-files-from-your-git-repo
4. If you have merge conflicts with your package-lock.json, use https://www.npmjs.com/package/npm-merge-driver. PLEASE DOWNLOAD THIS FROM THE VERY BEGINNING.
5. If you have an error with workerSrc:
  A. Go to line 8766 of node_modules/pdf.js-extract/lib/pdfjs/pdf.js
  B. Replace the function with this:
  ```javascript
  function getWorkerSrc() {
    return './node_modules/pdf.js-extract/lib/pdfjs/pdf.worker.js'
    // if (_worker_options.GlobalWorkerOptions.workerSrc) {
    //   return _worker_options.GlobalWorkerOptions.workerSrc;
    // }
    // if (typeof fallbackWorkerSrc !== 'undefined') {
    //   return fallbackWorkerSrc;
    // }
    // throw new Error('No "GlobalWorkerOptions.workerSrc" specified.');
  }
  ```
