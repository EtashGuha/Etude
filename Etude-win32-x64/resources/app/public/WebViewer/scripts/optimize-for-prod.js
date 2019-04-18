
const util = require('util');
const exec = util.promisify(require('child_process').exec);
const path = require('path');

const GREEN = "\x1b[32m";
const CYAN = "\x1b[36m";
const MAGENTA = "\x1b[35m";
const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const BLINK = "\x1b[5m";
const UNDER = "\x1b[4m";

const globalPrompt = {
  message: "Answer must be 'y' or 'n'",
  pattern: /^[yn]$/,
  required: true,
  type: 'string'
};

(async () => {
  try {
    require.resolve('prompt');
    require.resolve('fs-extra');
  } catch (e) {
    console.log(CYAN, `Installing required dependencies...`, RESET);
    await exec(`npm i prompt --save-dev && npm i fs-extra --save-dev`);
  }

  console.log(CYAN, `\nThis script will delete any files you won't be using in your lib folder. Please use with caution!`)
  console.log(CYAN, `\nPress CTRL + C at any time to safely cancel this process. If you are unsure of any anwsers, ${UNDER}please clarify${RESET}${CYAN} before answering them.`, RESET);

  const prompt = require('prompt');
  const fs = require('fs-extra');

  prompt.start();
  prompt.message = `${MAGENTA}\nOptimize`;
  prompt.delimiter = `: ${RESET}`;

  const backupExists = await fs.pathExists(path.resolve(__dirname, '../lib-backup'));
  if (backupExists) {
    console.log(CYAN, `\nA backup will not be created because a backup already exists!`);
  }

  const schema = {
    properties: {
      backup: {
        description: "Do you want us to backup your files before optimizing? [y/n]" + RESET,
        ask: () => {
          return !backupExists;
        },
        ...globalPrompt 
      },
      ui: {
        description: "Will you be using the new UI? [y/n]" + RESET,
        ...globalPrompt 
      },
      xod: {
        description: `Will you be converting all your documents to XOD? See ${CYAN}https://www.pdftron.com/documentation/web/guides/optimize-lib-folder${RESET}${DIM} for more info. [y/n]` + RESET,
        ...globalPrompt 
      },
      office: {
        description: "Do you need client side office viewing support? [y/n]" + RESET,
        ...globalPrompt,
        ask: () => {
          return prompt.history('xod').value === 'n'
        }
      },
      type: {
        description: `Do you need the full PDF API? See ${CYAN}https://www.pdftron.com/documentation/web/guides/optimize-lib-folder${RESET}${DIM} for more info (most users dont need this option). [y/n]` + RESET,
        ...globalPrompt,
        ask: () => {
          return prompt.history('xod').value === 'n'
        }
      }
    }
  }

  prompt.get(schema, (err, result) => {

    if (err) {
      console.log(`\n\n${RED}Process exited. No action will be taken.${RESET}\n`)
      return;
    }

    const { ui, xod, office, type, backup = 'n' } = result;

    let filesToDelete = [
      path.resolve(__dirname, '../lib/webviewer.js')
    ];

    // if they are using the new UI
    if (ui === 'y') {
      filesToDelete = [
        ...filesToDelete,
        path.resolve(__dirname, '../lib/ui-legacy'),
        path.resolve(__dirname, '../lib/ui/assets'),
        path.resolve(__dirname, '../lib/ui/CONTRIBUTING.md'),
        path.resolve(__dirname, '../lib/ui/dev-server.js'),
        path.resolve(__dirname, '../lib/ui/i18n'),
        path.resolve(__dirname, '../lib/ui/LICENSE'),
        path.resolve(__dirname, '../lib/ui/.babelrc'),
        path.resolve(__dirname, '../lib/ui/.eslintrc'),
        path.resolve(__dirname, '../lib/ui/package.json'),
        path.resolve(__dirname, '../lib/ui/README.md'),
        path.resolve(__dirname, '../lib/ui/src'),
        path.resolve(__dirname, '../lib/ui/webpack.config.dev.js'),
        path.resolve(__dirname, '../lib/ui/webpack.config.prod.js'),
      ];
    }
    // If they are using the OLD UI
    else {
      filesToDelete.push(
        path.resolve(__dirname, '../lib/ui')
      );
    }
  
    // If they are not using XOD
    if (xod === 'n') {

      // if they dont need office
      if (office === 'n') {
        filesToDelete.push(
          path.resolve(__dirname, '../lib/core/office')
        );
      }
  
      // If they dont need the full api
      if (type === 'n') {
        filesToDelete.push(
          path.resolve(__dirname, '../lib/core/pdf/full')
        );
      }
      // If they do need the full API
      else {
        filesToDelete.push(
          path.resolve(__dirname, '../lib/core/pdf/lean')
        );
      }
    }
    // if they are using XOD
    else {
      filesToDelete.push(
        path.resolve(__dirname, '../lib/core/office')
      );
      filesToDelete.push(
        path.resolve(__dirname, '../lib/core/pdf')
      );
    }

    console.log(`\n==== ${RED}${BLINK + UNDER}FILES & FOLDERS TO DELETE${RESET} ====\n`)

    filesToDelete.forEach(f => {
      console.log(`${RED}${f}${RESET}`);
    })

    console.log('\n===================================')

    prompt.get({
      properties: {
        delete: {
          description: `The above files will be permanently deleted. Is this okay? ${backup === 'y' ? "(A backup will be created in './lib-backup')" : "(A backup will NOT be created)"} [y|n]${RESET}`,
          ...globalPrompt,
        }
      }
    }, async (err, result) => { 

      if (err) {
        console.log(`\n\n${RED}Process exited. No action will be taken.${RESET}\n`)
        return;
      }

      if (result.delete === 'y') {

        if (backup === 'y') {
          console.log(`\n${GREEN}Creating backup...${RESET}`);
          await fs.copy(
            path.resolve(__dirname, '../lib'),
            path.resolve(__dirname, '../lib-backup'),
          )
        }

        console.log(`\n${GREEN}Deleting files...${RESET}`);

        const promises = filesToDelete.map(file => {
          return fs.remove(file);
        })

        await Promise.all(promises);

      } else {
        console.log(`\n${RED}Process exited. No action will be taken.${RESET}\n`)
        return;
      }

      console.log(`\n${GREEN}${UNDER}Done! Your lib folder is now optimized for production use.${RESET}\n\n`);
    })

  })
})()