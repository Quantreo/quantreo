document.addEventListener("DOMContentLoaded", function() {
    let sidebarInner = document.querySelector(".md-sidebar__inner");
    if (sidebarInner) {
        let imgDiv = document.createElement("div");
        imgDiv.classList.add("sidebar-image");
        imgDiv.innerHTML = '<img src="/assets/backtesting_thumbnail.png" style="width: 80%; border-radius: 10px; margin-top: 50px; display: block;">';
        sidebarInner.appendChild(imgDiv);  // Insère l'image à la fin, au lieu du début
    }
});
