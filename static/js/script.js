

//  <script type="text/javascript">
// var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
var $SCRIPT_ROOT = "http://127.0.0.1:8000/"
       
var intervalID = null;
var video_duration = 1000000
var current_time = 0
var is_running_stream=false;
var loadingStartStopButton=false;
var videoInitialized=false;
var objectDetectionList=[];
var loadingDetectionModel=false;
var selected_model_name=null;


 

function initVideoStreamFrame(){
    $('#videoFrame').attr("src", "video_stream");
    videoInitialized=true;
}

function initData(){
    getObjectDetectionList();
    if (videoInitialized==false)
        initVideoStreamFrame()

    // src="{{ url_for('video_stream') }}"
}

function onClickReset(){

}

function toggleStopStart(){        
    if(is_running_stream){
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(false);
        toggleDisabledLoadingModelButton(false);
        sendStopVideoRequest();

    }else{
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(true);
        toggleDisabledLoadingModelButton(true,showSpinner=false);
        sendStartVideoRequest()
    }
    is_running_stream=!is_running_stream
    
}

function fillobjectDetectionSelect(methodsList){
    var objectDetectionSelect = $("#objectDetectionSelect");
    console.log(methodsList);
    methodsList.forEach(method => {
        var el = document.createElement("option");
        el.textContent = method.name;
        el.value = method.name;
        objectDetectionSelect.append(el);
    });
}

function getObjectDetectionList(){
        $.ajax({
            type: "GET",
            url: $SCRIPT_ROOT + '/get_object_detection_list',
            dataType: "json",
            success: function (data) {
                // console.log(" get_object_detection_list_select")
                // console.log(data);
                // objectDetectionList=data;
                fillobjectDetectionSelect(data);
            },
            error: function (errMsg) {
                console.log(" ERROR IN get_object_detection_list")
            }
        });   
    }


function onClickLoadModel(){
    toggleDisabledLoadingModelButton(true);

    if (selected_model_name==null){
        selected_model_name=$( "#objectDetectionSelect" )[0].value
    }
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/models/load/'+selected_model_name,
        dataType: "json",
        success: function (data) {
            console.log(" /models/load/"+selected_model_name)
            // clearInterval(intervalID);
            toggleDisabledLoadingModelButton(false);
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
        }
    });
}

function sendStopVideoRequest(){
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/stop_stream',
        dataType: "json",
        success: function (data) {
            console.log(" stop_stream")
            // clearInterval(intervalID);
            $('#startStopButton').html( 'Start');
            $('#startStopButton').removeClass("btn-danger");
            $('#startStopButton').addClass("btn-success");
            toggleDisabledStartStopButton(false);
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
            toggleDisabledStartStopButton(false);
        }
    });
}
function sendStartVideoRequest(){
    
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_stream',
        dataType: "json",
        success: function (data) {
            console.log(" start_stream")
           
            // intervalID = setInterval(update_values, 600);
            $('#startStopButton').html( 'Stop');
            // $('#startStopButton').attr("class","btn btn-danger btn-lg w-25");
            $('#startStopButton').removeClass("btn-success");
            $('#startStopButton').addClass("btn-danger");
            toggleDisabledStartStopButton(false);
            // return data
        },
        error: function (errMsg) {
            console.log(" ERROR IN start_stream")
            toggleDisabledStartStopButton(false)
        }
    });    
}

 
function toggleDisabledDetectionMethodSelect(setToDisabled){
    if(setToDisabled){
        $("#objectDetectionSelect").prop('disabled', true);
    }else{
        $("#objectDetectionSelect").prop('disabled', false);
    }
}




function onChangeObjectDetection(){
    console.log($( "#objectDetectionSelect" )[0].value );
    selected_model_name=$( "#objectDetectionSelect" )[0].value
    
}

function toggleDisabledStartStopButton(setToDisabled){
    if (setToDisabled){
        $("#startStopButton").attr("disabled", true);
        loadingStartStopButton=true;
    }else{
        $("#startStopButton").attr("disabled", false);
        loadingStartStopButton=false;
    }
}

function toggleDisabledLoadingModelButton(setToDisabled,showSpinner=true){
    if (setToDisabled){
        $("#loadModelButton").attr("disabled", true);
        if(showSpinner){
            $("#loadModelButton").children().css( "display", "inline-block" )
        }
        loadingDetectionModel=true;
    }else{
        $("#loadModelButton").attr("disabled", false);
        $("#loadModelButton").children().css( "display", "none" )
        loadingDetectionModel=false;
    }
}

function update_values() {
    $.getJSON($SCRIPT_ROOT + '/current_time',
        function (data) {
            if (data) {
                current_time = data.result
                $('#myRange').val(data.result)
            } else {
                clearInterval(intervalID);
            }
        });

    if (Math.abs(video_duration - current_time) < 0.1) {
        clearInterval(intervalID);
    }
};

function load_duration() {
    $.getJSON($SCRIPT_ROOT + '/video_duration',
        function (data) {

            video_duration = data.result
            $('#myRange').attr("max", data.result)

        });
}



window.onbeforeunload = function (e) {
    var e = e || window.event;

    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/clean_memory',
        // The key needs to match your method's input parameter (case-sensitive).
        // data: JSON.stringify({ Markers: markers }),
        // contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            console.log(" memory cleaned")
            // alert("")
            return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN memory cleaning")
        }
    });


};
//  </script>