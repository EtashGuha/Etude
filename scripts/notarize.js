require('dotenv').config();
const { notarize } = require('@mceachen/electron-notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;
  if (electronPlatformName !== 'darwin') {
    return;
  }

  const appName = context.packager.appInfo.productFilename;

  return await notarize({
    appBundleId: 'com.github.EtashGuha.Etude',
    appPath: `${appOutDir}/${appName}.app`,
    appleId: "etashguha@icloud.com",
    appleIdPassword: "szwu-efkb-vhal-tdvo",
  });
};
