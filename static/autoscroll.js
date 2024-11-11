const iframe = document.getElementById("plp-frame");

  iframe.onload = function () {
    // Use postMessage to send scroll command to the PDF viewer
    iframe.contentWindow.postMessage({ type: 'scroll', scrollTo: 'end' }, '*');
  };