document.addEventListener('DOMContentLoaded', function() {
    // Theme toggle logic
    const themeToggle = document.getElementById('theme-toggle');
    const setTheme = (mode) => {
        document.body.classList.toggle('light-mode', mode === 'light');
        document.body.classList.toggle('dark-mode', mode === 'dark');
        if (themeToggle) {
            themeToggle.innerHTML = mode === 'dark' ? '🌙' : '☀️';
            themeToggle.setAttribute('aria-label', mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
        }
    };
    // Default to dark mode
    let theme = localStorage.getItem('theme') || 'dark';
    setTheme(theme);
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            theme = (theme === 'dark') ? 'light' : 'dark';
            setTheme(theme);
            localStorage.setItem('theme', theme);
        });
    }
    // Recommendation form logic
    const form = document.getElementById('recommend-form');
    const recommendationsDiv = document.getElementById('recommendations');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            recommendationsDiv.innerHTML = '<em>Loading recommendations...</em>';
            const use_case = document.getElementById('use-case').value;
            const max_budget = document.getElementById('max-budget').value;
            const preferred_brand = document.getElementById('brand').value;
            fetch('/api/recommend/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ use_case, max_budget, preferred_brand })
            })
            .then(res => res.json())
            .then(data => {
                let html = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    html += '<h3>Top Recommendations:</h3>';
                    html += '<ol>';
                    data.recommendations.forEach(lap => {
                        html += `<li><strong>${lap.Laptop}</strong> <br>Brand: ${lap.Brand} <br>CPU: ${lap.CPU} <br>RAM: ${lap.RAM}GB <br>Storage: ${lap.Storage}GB ${lap.Storage_type || ''} <br>GPU: ${lap.GPU} <br>Screen: ${lap.Screen} <br>Price: $${lap.Price.toFixed(2)} <br>Best for: ${lap.Category}</li><hr>`;
                    });
                    html += '</ol>';
                }
                if (data.similar && data.similar.length > 0) {
                    html += '<div class="similar-block"><h3>Similar Laptops:</h3><ol>';
                    data.similar.forEach(lap => {
                        html += `<li><strong>${lap.Laptop}</strong> <br>Brand: ${lap.Brand} <br>CPU: ${lap.CPU} <br>RAM: ${lap.RAM}GB <br>Storage: ${lap.Storage}GB ${lap.Storage_type || ''} <br>GPU: ${lap.GPU} <br>Screen: ${lap.Screen} <br>Price: $${lap.Price.toFixed(2)} <br>Best for: ${lap.Category}</li><hr>`;
                    });
                    html += '</ol></div>';
                }
                if (data.error) {
                    html = `<span style='color:red;'>${data.error}</span>`;
                } else if (!data.recommendations || data.recommendations.length === 0) {
                    html = '<span>No recommendations found.</span>';
                }
                recommendationsDiv.innerHTML = html;
            })
            .catch(err => {
                recommendationsDiv.innerHTML = `<span style='color:red;'>Error: ${err}</span>`;
            });
        });
    }
}); 