const imageInput = document.getElementById('image'); 
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image'); 
const clearImageButton = document.getElementById('clear-image'); 
const resultsDiv = document.getElementById('results'); 
const titleInput = document.getElementById('title'); 

imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result; 
            previewContainer.style.display = 'flex'; 
        };
        reader.readAsDataURL(file);
    } else {
        previewContainer.style.display = 'none'; 
    }
});

clearImageButton.addEventListener('click', async function () {
    imageInput.value = ''; 
    previewImage.src = ''; 
    previewContainer.style.display = 'none'; 
    resultsDiv.innerHTML = ''; 
    titleInput.value = ''; 

    try {
        const response = await fetch('/reset', { method: 'POST' });
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        console.log("State successfully reset.");
    } catch (error) {
        console.error("Error resetting state:", error.message);
    }
});

document.getElementById('search-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const title = titleInput.value.trim();
    const image = imageInput.files[0];

    if (!title && !image) {
        resultsDiv.innerHTML = `<p style="color: red;">Please enter a title or choose an image for the search.</p>`;
        return;
    }

    if (title) formData.append('title', title);
    if (image) formData.append('image_path', image);

    resultsDiv.innerHTML = `<p>Searching...</p>`;

    try {
        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.results && data.results.length > 0) {
            resultsDiv.innerHTML = data.results.map(book => `
                <div class="book">
                    <h3>${book.title}</h3>
                    <p><strong>Price:</strong> $${book.price}</p>
                    <p><strong>Relevance:</strong> ${book.relevance_score.toFixed(2)}</p>
                    ${book.encoded_image ? `<img src="data:image/jpeg;base64,${book.encoded_image}" alt="${book.title}" style="max-width: 100%; border-radius: 5px;">` : ''}
                </div>
            `).join('');
        } else {
            resultsDiv.innerHTML = `<p>No results found.</p>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});
