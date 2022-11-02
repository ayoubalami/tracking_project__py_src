

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
        setToStopButton()
    }else{
        setToStartButton()
    }
    is_running_stream=!is_running_stream
    // console.log(startStopButton)
    // $("#startStopButton").html('Save');
    // alert(startStopButton.prop('value'))
}

function fillobjectDetectionSelect(methodsList){
    var objectDetectionSelect = $("#objectDetectionSelect");
    methodsList.forEach(method => {
        var el = document.createElement("option");
        el.textContent = method;
        el.value = method;
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


function setToStopButton(){

    disabledStartStopButton();
    enableDetectionMethodSelect();
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
            enabledStartStopButton()
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
            enabledStartStopButton()
        }
    });
}
function setToStartButton(){
    disabledDetectionMethodSelect()
    disabledStartStopButton()
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
            enabledStartStopButton()
            // return data
        },
        error: function (errMsg) {
            console.log(" ERROR IN start_stream")
            enabledStartStopButton()
        }
    });    
}

function disabledDetectionMethodSelect(){
    $("#objectDetectionSelect").prop('disabled', true);
    console.log("disabledDetectionMethodSelect");
}

function enableDetectionMethodSelect(){
    $("#objectDetectionSelect").prop('disabled', false);
}

function disabledStartStopButton(){
    $("#startStopButton").attr("disabled", true);
    loadingStartStopButton=true;
}

function enabledStartStopButton(){
    $("#startStopButton").attr("disabled", false);
    loadingStartStopButton=false;

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