

function reloadPDF() {
    // Get the current time and append it as a query string to bypass the cache
    var plpFrame = document.getElementById('plp-frame');
    plpFrame.src = file + '?t=' + new Date().getTime().toString();
    simulateScroll();
    const status = fetch(file + '/status').then(x => x.text());
    if (status == 'done') {
      clearInterval(id)
    }
}

function simulateScroll() {
  const rect = iframe.getBoundingClientRect();
  const width = rect.right - rect.left;
  const height = rect.bottom - rect.top;
  const x = rect.left + width / 2;
  const y = rect.top + height / 2;
  const deltaY = (iframe.contentDocument || iframe.contentWindow.document).body.scrollHeight
  const event = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: deltaY, // Positive for scroll down, negative for scroll up'
      clientX: x,
      clientY: y
  });

  // Dispatch the event on the document or a specific element
  document.dispatchEvent(event);
}

const iframe = document.getElementById("plp-frame");

function scroll() {
    // Use postMessage to send scroll command to the PDF viewer
    iframe.contentWindow.postMessage({ type: 'scroll', scrollTo: 'end' }, '*');
  };
  iframe.onload = scroll

const id = setInterval(reloadPDF, 5000);  // Refresh every 5 seconds (5000 ms)

reloadPDF()