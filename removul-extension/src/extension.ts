
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as child_process from 'child_process';

interface HighlightedRange {
  documentUri: vscode.Uri;
  lineNumbers: number[];
}

let highlightedRanges: HighlightedRange[] = [];

export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('removul.helloWorld', () => {
    const folderUri = vscode.workspace.workspaceFolders?.[0].uri;
    if (!folderUri) {
      vscode.window.showErrorMessage('No workspace folder found.');
      return;
    }

    const directoryPath = folderUri.fsPath;
    const pythonScriptPath = '/home/raghad/Desktop/GP/removul/removul.py';
    const pythonCommand = `. /home/raghad/Desktop/GP/removul/removul/bin/activate && python3 "${pythonScriptPath}" --directory_name "${directoryPath}"`;

    vscode.window.showInformationMessage(pythonCommand);
    child_process.exec(pythonCommand, (error, stdout, stderr) => {
      if (error) {
        vscode.window.showErrorMessage('Error executing Python script: ' + error);
        return;
      }

      try {
        vscode.window.showInformationMessage(stdout);
      }
      catch (parseError) {
        vscode.window.showErrorMessage('Error parsing Python script output: ' + parseError);
      }
    });

    // fs.readdir(directoryPath, (err, files) => {
    //   if (err) {
    //     vscode.window.showErrorMessage('Error reading directory: ' + err);
    //     return;
    //   }

    //   files.forEach((file) => {
    //     const filePath = path.join(directoryPath, file);
    //     const pythonScriptPath = '/home/raghad/Desktop/GP/removul/removul.py';
    //     const pythonCommand = `. /home/raghad/Desktop/GP/removul/removul/bin/activate && python3 "${pythonScriptPath}" --file_name "${filePath}"`;

    //     child_process.exec(pythonCommand, (error, stdout, stderr) => {
    //       if (error) {
    //         vscode.window.showErrorMessage('Error executing Python script: ' + error);
    //         return;
    //       }

    //       try {
    //         const lineNumbers_func = stdout.trim().split('\n');
            
    //         for (var i = 0; i < lineNumbers_func.length; i++) {
    //           const lineNumbers = lineNumbers_func[i].split(',').map(Number);
            
    //           if (lineNumbers.length > 0) {
    //             const documentUri = vscode.Uri.file(filePath);
    //             vscode.workspace.openTextDocument(documentUri).then((doc) => {
    //               vscode.window.showTextDocument(doc, vscode.ViewColumn.One).then(() => {
    //                 //const highlightedRangesForFile: HighlightedRange[] = [];
    //                 lineNumbers.forEach((lineNumber, index) => {
    //                   var opacity = 1 - index / (lineNumbers.length+1);
    //                   const decoration = vscode.window.createTextEditorDecorationType({
    //                     backgroundColor: `rgba(255, 0, 0, ${opacity})`
    //                   });
                      
    //                   //highlightedRangesForFile.push(highlightedRange);
    //                   const range = new vscode.Range(lineNumber - 1, 0, lineNumber - 1, 100);
    //                   vscode.window.activeTextEditor?.setDecorations(decoration, [{ range }]);
    //                 });
    //                 const highlightedRange: HighlightedRange = {
    //                   documentUri: doc.uri,
    //                   lineNumbers: lineNumbers,
    //                 };
    //                 highlightedRanges.push(highlightedRange);
    //               });
    //             });
    //           }
    //         }
    //       } catch (parseError) {
    //         vscode.window.showErrorMessage('Error parsing Python script output: ' + parseError);
    //       }
    //     });
    //   });
    // });
  });

  context.subscriptions.push(disposable);

  // Reapply highlights when changing active editor tab
  vscode.window.onDidChangeActiveTextEditor((editor) => {
    if (editor) {
      const editorUri = editor.document.uri;
      const highlightedRange = highlightedRanges.find(
        (range) => range.documentUri.toString() === editorUri.toString()
      );
      if (highlightedRange) {
        highlightedRange.lineNumbers.forEach((lineNumber, index) => {
          var opacity = 1 - index / (highlightedRange.lineNumbers.length+1);
          const decoration = vscode.window.createTextEditorDecorationType({
            backgroundColor: `rgba(255, 0, 0, ${opacity})`
          });
          
          //highlightedRangesForFile.push(highlightedRange);
          const range = new vscode.Range(lineNumber - 1, 0, lineNumber - 1, 100);
          vscode.window.activeTextEditor?.setDecorations(decoration, [{ range }]);
        });
      }
    }
  });
}

export function deactivate() {}

// import * as vscode from 'vscode';
// import * as fs from 'fs';
// import * as path from 'path';
// import * as child_process from 'child_process';

// interface HighlightedRange {
//   documentUri: vscode.Uri;
//   lineNumbers: number[];
//   decoration: vscode.TextEditorDecorationType;
// }

// let highlightedRanges: HighlightedRange[] = [];

// export function activate(context: vscode.ExtensionContext) {
//   let disposable = vscode.commands.registerCommand('removul.helloWorld', () => {
//     const folderUri = vscode.workspace.workspaceFolders?.[0].uri;
//     if (!folderUri) {
//       vscode.window.showErrorMessage('No workspace folder found.');
//       return;
//     }

//     const directoryPath = folderUri.fsPath;
//     fs.readdir(directoryPath, (err, files) => {
//       if (err) {
//         vscode.window.showErrorMessage('Error reading directory: ' + err);
//         return;
//       }

//       files.forEach((file) => {
//         const filePath = path.join(directoryPath, file);
//         const pythonScriptPath = '/home/raghad/Desktop/GP/removul/code/lines.py'; //path.join(context.extensionPath, 'extract_lines.py');
//         const pythonCommand = `python "${pythonScriptPath}" "${filePath}"`;

//         child_process.exec(pythonCommand, (error, stdout, stderr) => {
//           if (error) {
//             vscode.window.showErrorMessage('Error executing Python script: ' + error);
//             return;
//           }

//           try {
//             const lineNumbers = stdout.trim().split('\n').map(Number);

//             if (lineNumbers.length > 0) {
//               const documentUri = vscode.Uri.file(filePath);
//               vscode.workspace.openTextDocument(documentUri).then((doc) => {
//                 vscode.window.showTextDocument(doc, vscode.ViewColumn.One).then(() => {
//                   const decoration = vscode.window.createTextEditorDecorationType({
//                     backgroundColor: 'Red',
//                   });
//                   const highlightedRange: HighlightedRange = {
//                     documentUri: doc.uri,
//                     lineNumbers,
//                     decoration,
//                   };
//                   highlightedRanges.push(highlightedRange);
//                   const ranges = highlightedRange.lineNumbers.map((lineNumber) =>
//                   new vscode.Range(lineNumber - 1, 0, lineNumber - 1, 100)
//                   );        
//                   vscode.window.activeTextEditor?.setDecorations(createDensityDecoration(lineNumbers.length), ranges);
//                 });
//               });
//             }
//           } catch (parseError) {
//             vscode.window.showErrorMessage('Error parsing Python script output: ' + parseError);
//           }
//         });
//       });
//     });
//   });

//   context.subscriptions.push(disposable);

//   // Reapply highlights when changing active editor tab
//   vscode.window.onDidChangeActiveTextEditor((editor) => {
//     if (editor) {
//       const editorUri = editor.document.uri;
//       const highlightedRange = highlightedRanges.find(
//         (range) => range.documentUri.toString() === editorUri.toString()
//       );
//       if (highlightedRange) {
//         const ranges = highlightedRange.lineNumbers.map((lineNumber) =>
//           new vscode.Range(lineNumber - 1, 0, lineNumber - 1, 100)
//         );
//         editor.setDecorations(highlightedRange.decoration, ranges);
//       }
//     }
//   });
// }

// function createDensityDecoration(lineCount: number): vscode.TextEditorDecorationType {
//   const colors: string[] = [];
//   const opacityStep = 1 / lineCount;
//   for (let i = 0; i < lineCount; i++) {
//     const opacity = 1 - (i * opacityStep);
//     colors.push(`rgba(255, 0, 0, ${opacity})`);
//   }
//   return vscode.window.createTextEditorDecorationType({ backgroundColor: colors });
// }

// export function deactivate() {}


// import * as vscode from 'vscode';
// import * as fs from 'fs';
// import * as path from 'path';

// interface HighlightedRange {
//   documentUri: vscode.Uri;
//   range: vscode.Range;
//   decoration: vscode.TextEditorDecorationType;
// }

// let highlightedRanges: HighlightedRange[] = [];

// export function activate(context: vscode.ExtensionContext) {
//   let disposable = vscode.commands.registerCommand(
//     'removul.helloWorld',
//     () => {
//       const folderUri = vscode.workspace.workspaceFolders?.[0].uri;
//       if (!folderUri) {
//         vscode.window.showErrorMessage('No workspace folder found.');
//         return;
//       }

//       const directoryPath = folderUri.fsPath;
//       fs.readdir(directoryPath, (err, files) => {
//         if (err) {
//           vscode.window.showErrorMessage('Error reading directory: ' + err);
//           return;
//         }

//         files.forEach((file) => {
//           const filePath = path.join(directoryPath, file);
//           fs.readFile(filePath, 'utf-8', (err, data) => {
//             if (err) {
//               vscode.window.showErrorMessage(
//                 'Error reading file: ' + filePath + ' - ' + err
//               );
//               return;
//             }

//             const lines = data.split('\n');
//             const thirdLine = lines[2];

//             if (thirdLine) {
//               const documentUri = vscode.Uri.file(filePath);
//               vscode.workspace.openTextDocument(documentUri).then((doc) => {
//                 vscode.window.showTextDocument(doc, vscode.ViewColumn.One).then(() => {
//                   const range = new vscode.Range(
//                     new vscode.Position(2, 0), // Third line, zero-based index
//                     new vscode.Position(2, thirdLine.length) // End of the line
//                   );
//                   const decoration = vscode.window.createTextEditorDecorationType({
//                     backgroundColor: 'yellow',
//                   });
//                   const highlightedRange: HighlightedRange = {
//                     documentUri: doc.uri,
//                     range,
//                     decoration,
//                   };
//                   highlightedRanges.push(highlightedRange);
//                   vscode.window.activeTextEditor?.setDecorations(decoration, [range]);
//                 });
//               });
//             }
//           });
//         });
//       });
//     }
//   );

//   context.subscriptions.push(disposable);

//   // Reapply highlights when changing active editor tab
//   vscode.window.onDidChangeActiveTextEditor((editor) => {
//     if (editor) {
//       const editorUri = editor.document.uri;
//       const highlightedRange = highlightedRanges.find(
//         (range) => range.documentUri.toString() === editorUri.toString()
//       );
//       if (highlightedRange) {
//         editor.setDecorations(highlightedRange.decoration, [highlightedRange.range]);
//       }
//     }
//   });
// }

// export function deactivate() {}

//--------------------------------------------------------------------------------

// import * as vscode from 'vscode';
// import * as fs from 'fs';
// import * as path from 'path';

// // This function scans a directory and returns a list of all file paths
// function scanDirectory(directory: string): string[] {
//   let filepaths: string[] = [];
//   const files = fs.readdirSync(directory);
//   for (const file of files) {
//     const filePath = path.join(directory, file);
//     const stats = fs.statSync(filePath);
//     if (stats.isDirectory()) {
//       filepaths = filepaths.concat(scanDirectory(filePath));
//     } else {
//       filepaths.push(filePath);
//     }
//   }
//   return filepaths;
// }

// // This function searches for vulnerable lines in a file and highlights them
// function highlightVulnerableLines(document: vscode.TextDocument): void {
//   const text = document.getText();
//   // Implement your logic to identify vulnerable lines here
//   // You can use regular expressions or any other method suitable for your use case

//   // Example: Highlight all lines containing the word "vulnerable"
//   const lines = text.split('\n');
//   const vulnerableLines = lines.filter((line) => line.includes('vulnerable'));

//   const decorations = vulnerableLines.map((line) => ({
//     range: new vscode.Range(lines.indexOf(line), 0, lines.indexOf(line), line.length),
//     hoverMessage: 'This line is vulnerable',
//   }));

//   // Apply the decorations to the current document
//   vscode.window.activeTextEditor?.setDecorations(vulnerableLineDecorationType, decorations);
// }

// // Register the command and decoration type
// let vulnerableLineDecorationType: vscode.TextEditorDecorationType;

// export function activate(context: vscode.ExtensionContext): void {
//   console.log('Vulnerable Line Highlighter extension activated');

//   // Create a decoration type
//   vulnerableLineDecorationType = vscode.window.createTextEditorDecorationType({
//     backgroundColor: 'rgba(255, 0, 0, 0.3)',
//   });

//   // Register a command that triggers the scanning and highlighting process
//   let disposable = vscode.commands.registerCommand('extension.highlightVulnerableLines', async () => {
//     const rootPath = vscode.workspace.rootPath;
//     if (rootPath) {
//       const filepaths = scanDirectory(rootPath);
//       for (const filepath of filepaths) {
//         const document = await vscode.workspace.openTextDocument(filepath);
//         await vscode.window.showTextDocument(document);
//         highlightVulnerableLines(document);
//       }
//     } else {
//       vscode.window.showErrorMessage('Please open a workspace folder');
//     }
//   });

//   context.subscriptions.push(disposable);
// }

// export function deactivate(): void {
//   console.log('Vulnerable Line Highlighter extension deactivated');
// }
