console.log("app.js loaded");

window.addEventListener("beforeunload", () => {
    console.log("PAGE IS RELOADING");
});


const API_BASE = "https://smart-resume-screener-dc12.onrender.com";


let activeJobId = localStorage.getItem("activeJobId");



const jobTitleInput = document.getElementById("job-title");
const jobDescInput = document.getElementById("job-description");
const jobHint = document.getElementById("job-hint");

const jobForm = document.getElementById("job-form");
const activeJobBox = document.getElementById("active-job");
const activeJobTitle = document.getElementById("active-job-title");


const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadLog = document.getElementById("upload-log");


const candidateList = document.getElementById("candidate-list");
const emptyState = document.getElementById("empty-state");


const modal = document.getElementById("detail-modal");
const modalBody = document.getElementById("modal-body");




// ================= RESTORE ROLE =================


async function restoreJob(){

    if(!activeJobId)
        return;


    try{

        const res = await fetch(
            `${API_BASE}/jobs/${activeJobId}`
        );


        if(!res.ok){

            localStorage.removeItem("activeJobId");
            activeJobId = null;

            return;
        }


        const job = await res.json();


        console.log(
            "Restored job:",
            job
        );


        activeJobTitle.textContent =
            job.title;


        activeJobBox.classList.remove("hidden");

        jobForm.classList.add("hidden");


        loadCandidates();


    }
    catch(err){

        console.error(
            "Restore error:",
            err
        );

    }

}


restoreJob();




// ================= CREATE ROLE =================


document
.getElementById("create-job-btn")
.addEventListener(
"click",
async function(e){


    e.preventDefault();


    const title =
        jobTitleInput.value.trim();


    const description =
        jobDescInput.value.trim();



    if(!title || !description){

        jobHint.textContent =
        "Add title and description first.";

        return;

    }



    jobHint.textContent =
    "Creating role...";



    try{


        const res =
        await fetch(
        `${API_BASE}/jobs`,
        {

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify({

                title:title,
                description:description

            })

        });



        if(!res.ok){

            throw new Error(
                await res.text()
            );

        }



        const job =
        await res.json();



        console.log(
            "JOB CREATED:",
            job
        );



        activeJobId =
            job.id;



        localStorage.setItem(
            "activeJobId",
            job.id
        );



        activeJobTitle.textContent =
            job.title;



        jobForm.classList.add("hidden");

        activeJobBox.classList.remove("hidden");


        jobHint.textContent="";


        loadCandidates();



    }
    catch(err){


        jobHint.textContent =
        err.message;


    }


});





// ================= SWITCH ROLE =================


document
.getElementById("change-job-btn")
.addEventListener(
"click",
function(){


    activeJobId=null;


    localStorage.removeItem(
        "activeJobId"
    );


    activeJobBox.classList.add("hidden");


    jobForm.classList.remove("hidden");


    candidateList.innerHTML="";


    emptyState.classList.remove("hidden");


});





// ================= UPLOAD =================



dropzone.addEventListener(
"click",
function(){

    if(!activeJobId){

        alert(
        "Create a role first"
        );

        return;

    }


    fileInput.click();

});





fileInput.addEventListener(
"change",
function(e){


    const files =
        e.target.files;



    if(files.length){

        console.log(
            "Selected files:",
            files
        );


        handleFiles(files);

    }


    // allow selecting same file again

    e.target.value="";


});






dropzone.addEventListener(
"dragover",
function(e){

    e.preventDefault();

    dropzone.classList.add("drag");

});



dropzone.addEventListener(
"dragleave",
function(){

    dropzone.classList.remove("drag");

});



dropzone.addEventListener(
"drop",
function(e){


    e.preventDefault();


    dropzone.classList.remove("drag");


    handleFiles(
        e.dataTransfer.files
    );


});






function handleFiles(files){


    if(!activeJobId){

        alert(
        "Create a role first"
        );

        return;

    }



    [...files].forEach(
        uploadResume
    );

}





async function uploadResume(file){


    console.log(
        "Uploading:",
        file.name
    );



    const logItem =
        document.createElement("li");



    logItem.className =
        "status-pending";



    logItem.innerHTML =
    `
    <span>${file.name}</span>
    <span>Analyzing...</span>
    `;



    uploadLog.prepend(
        logItem
    );




    const formData =
        new FormData();



    formData.append(
        "file",
        file
    );



    try{


        const res =
        await fetch(
        `${API_BASE}/jobs/${activeJobId}/resumes`,
        {

            method:"POST",

            body:formData

        });



        console.log(
            "Upload response:",
            res.status
        );



        if(!res.ok){

            throw new Error(
                await res.text()
            );

        }



        const data =
            await res.json();



        console.log(
            "UPLOAD SUCCESS:",
            data
        );



        logItem.className =
            "status-ok";


        logItem.innerHTML =
        `
        <span>${file.name}</span>

        <span>
        Score ${data.match_score}/10
        </span>
        `;



        loadCandidates();



    }
    catch(err){


        console.error(
            "UPLOAD FAILED:",
            err
        );


        logItem.className =
            "status-error";


        logItem.innerHTML =
        `
        <span>${file.name}</span>

        <span>
        ${err.message}
        </span>
        `;


    }

}






// ================= CANDIDATES =================



document
.getElementById("refresh-btn")
.addEventListener(
"click",
loadCandidates
);




async function loadCandidates(){


    if(!activeJobId)
        return;



    try{


        const res =
        await fetch(
        `${API_BASE}/jobs/${activeJobId}/candidates`
        );



        const candidates =
            await res.json();



        renderCandidates(
            candidates
        );



    }
    catch(err){

        console.error(err);

    }

}





function renderCandidates(candidates){


    candidateList.innerHTML="";



    if(!candidates.length){

        emptyState.classList.remove("hidden");

        return;

    }



    emptyState.classList.add("hidden");



    candidates.forEach(c=>{


        const card =
        document.createElement("div");

        card.className =
        "candidate-card";



        card.innerHTML =
        `

        <div class="score-stamp">
        ${c.match_score ?? "-"}
        </div>


        <div class="candidate-info">

        <h4>
        ${c.candidate_name || "Unnamed"}
        </h4>


        <div class="filename">
        ${c.filename}
        </div>


        <p>
        ${c.justification || ""}
        </p>


        </div>

        `;



        candidateList.appendChild(card);


    });


}





// ================= MODAL =================


document
.getElementById("modal-close")
.addEventListener(
"click",
()=>modal.classList.add("hidden")
);