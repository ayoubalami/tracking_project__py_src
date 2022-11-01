

//  <script type="text/javascript">
// var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
var $SCRIPT_ROOT = "http://127.0.0.1:8000/"
       
var intervalID = null;
var video_duration = 1000000
var current_time = 0

var is_running_stream=false;


function toggleStopStart(){
    
    if(is_running_stream){
        setToStopButton()
        // alert(running_stream)
    }else{
        setToStartButton()
        // alert(running_stream)
    }
    is_running_stream=!is_running_stream

    // console.log(startStopButton)
    // $("#startStopButton").html('Save');
    // alert(startStopButton.prop('value'))
}


function stopStream(){
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/stop_stream',
        dataType: "json",
        success: function (data) {
            console.log(" stop_stream")
            // clearInterval(intervalID);
            return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
        }
    });
}

function startStream(){
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_stream',
        dataType: "json",
        success: function (data) {
            console.log(" start_stream")
            // intervalID = setInterval(update_values, 600);
            return data
        },
        error: function (errMsg) {
            console.log(" ERROR IN start_stream")
        }
    });
}

function setToStopButton(){
    stopStream();
    $('#startStopButton').html( 'Start');
    $('#startStopButton').attr("class","btn btn-success btn-lg w-25");
}

function setToStartButton(){
    startStream();
    $('#startStopButton').html( 'Stop');
    $('#startStopButton').attr("class","btn btn-danger btn-lg w-25");
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