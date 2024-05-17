window.addEventListener('DOMContentLoaded', event => {
    // Simple-DataTables
    // https://github.com/fiduswriter/Simple-DataTables/wiki

    const datatablesAlertes = document.getElementById('datatablesAlertes');
    if (datatablesAlertes) {
        new simpleDatatables.DataTable(datatablesAlertes);
    }

    const datatablesInfos = document.getElementById('datatablesInfos');
    if (datatablesInfos) {
        new simpleDatatables.DataTable(datatablesInfos);
    }
});
