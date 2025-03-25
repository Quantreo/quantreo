document.addEventListener("DOMContentLoaded", function() {
    let sidebarInner = document.querySelector(".md-sidebar__inner");
    if (sidebarInner) {
        let imgDiv = document.createElement("div");
        imgDiv.classList.add("sidebar-image");
        imgDiv.innerHTML = `
            <a href="https://www.quantreo.com" target="_blank" style="text-decoration: none;">
                <img src="/assets/newsletter.webp" class="sidebar-img" style="width: 80%; border-radius: 10px; margin-top: 50px; display: block; transition: transform 0.3s ease;">
            </a>
        `;
        sidebarInner.appendChild(imgDiv);

        // Ajouter le style hover dynamiquement via JS
        let style = document.createElement('style');
        style.innerHTML = `
            .sidebar-img:hover {
                transform: scale(1.02);
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }
});
