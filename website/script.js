const copyButton = document.getElementById('copy');
if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById('commands').textContent);
      copyButton.textContent = 'Copied!';
      status.textContent = 'Quick start commands copied to clipboard.';
      setTimeout(() => { copyButton.textContent = 'Copy commands'; }, 2000);
    } catch {
      status.textContent = 'Could not copy. Select the commands and copy them manually.';
      copyButton.textContent = 'Select text to copy';
    }
  });
}
