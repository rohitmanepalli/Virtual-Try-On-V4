const path = require('node:path');
const { app, BrowserWindow, session } = require('electron');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const fs = require('node:fs');
const isDev = process.env.ELECTRON_DEV === '1';
const API_BASE_URL = (process.env.VITE_API_BASE_URL || 'http://localhost:8888').replace(/\/+$/, '');
const DEV_SERVER_URL = 'http://localhost:5173';
const distAppIndex = path.join(__dirname, '..', 'dist-app', 'index.html');
const distIndex = path.join(__dirname, '..', 'dist', 'index.html');
const APP_INDEX = fs.existsSync(distAppIndex) ? distAppIndex : distIndex;

const HEALTH_CHECK_TIMEOUT_MS = 3000;
const HEALTH_CHECK_RETRY_DELAY_MS = 2000;

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    show: false,
    fullscreen: !isDev,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media');
  });

  mainWindow.loadFile(path.join(__dirname, 'splash.html'));
  mainWindow.once('ready-to-show', () => mainWindow.show());

  mainWindow.webContents.on('did-finish-load', () => {
    if (mainWindow.webContents.getURL().includes('splash.html')) {
      waitForBackend();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function sendSplashStatus(text) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('splash-status', text);
  }
}

async function checkBackend() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);
  try {
    await fetch(`${API_BASE_URL}/`, {
      signal: controller.signal,
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function waitForBackend() {
  sendSplashStatus('Connecting to server…');
  const reachable = await checkBackend();
  if (reachable) {
    loadApp();
    return;
  }
  sendSplashStatus("Can't reach server, retrying…");
  setTimeout(waitForBackend, HEALTH_CHECK_RETRY_DELAY_MS);
}

function loadApp() {
  if (isDev) {
    mainWindow.loadURL(DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(APP_INDEX);
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
