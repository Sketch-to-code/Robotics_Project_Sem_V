const scanBtn = document.getElementById('scan-btn');
const scanPreview = document.getElementById('scan-preview');

scanBtn.addEventListener('click', () => {
    scanPreview.innerHTML = '<span class="placeholder-text">Scanning...</span>';

    fetch('/api/scan')
        .then(response => response.json())
        .then(data => {
            // Build a small readable line for each detected cube: "#0 at (120, 340)"
            const cubeLines = data.cubes.map(cube =>
                `#${cube.id} at (${cube.centroid_px[0]}, ${cube.centroid_px[1]})`
            );

            // .join('<br>') puts each cube on its own line inside the preview box
            scanPreview.innerHTML = `
                <div>
                    <strong>${data.cubes_found} cubes found</strong><br>
                    ${cubeLines.join('<br>')}
                </div>
            `;
        })
        .catch(error => {
            scanPreview.innerHTML = '<span class="placeholder-text">Scan failed</span>';
            console.error('Scan error:', error);
        });
});