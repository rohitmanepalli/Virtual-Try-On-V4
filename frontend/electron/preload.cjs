const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('splashBridge', {
  onStatus: (callback) => {
    ipcRenderer.on('splash-status', (_event, text) => callback(text));
  },
});
