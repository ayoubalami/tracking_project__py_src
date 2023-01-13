

//  <script type="text/javascript">
// var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};

// var $SCRIPT_ROOT = "http://Raspberrypi.local:8000/"
var $SCRIPT_ROOT = "http://"+api_server+":8000"
var secondApiIpChecked=false

var intervalID = null;
var video_duration = 1000000
var current_time = 0
var is_running_stream=false;
var loadingStartStopButton=false;
var videoInitialized=false;
var objectDetectionList=[];
var loadingDetectionModel=false;
var selected_model_name=null;
var showBackgroundSubtractionStream=false;

window.onbeforeunload = function(event){

    console.log("loading ....");
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/clean_memory',
            dataType: "json",
            success: function (data) {
                console.log(" memory cleaned")
                // alert("")
                return true
            },
            error: function (errMsg) {
                console.log(" ERROR IN memory cleaning")
            }
        });    
        // return true    
        // return confirm("Confirm refresh");
    }


    function updateThresholdValue(){
        var thresholdValueText =  $( "#thresholdValueText" )
        var newThresholdValue=$( "#thresholdSlider" )[0].value;
        thresholdValueText.text( newThresholdValue);
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/models/update_threshold/'+newThresholdValue,
            dataType: "json",
            success: function (data) {
                console.log("/models/threshold")
                
            },
            error: function (errMsg) {
            }
        });  
    }

    function updateNmsThresholdValue(){
       
        var nmsThresholdValueText =  $( "#nmsThresholdValueText" )
        var newNmsThresholdValue=$( "#nmsThresholdSlider" )[0].value;

        nmsThresholdValueText.text( newNmsThresholdValue);
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/models/update_nms_threshold/'+newNmsThresholdValue,
            dataType: "json",
            success: function (data) {
                console.log("/models/nms_threshold") 
            },
            error: function (errMsg) {
            }
        });  
    }

    function updateBackgroundSubtractionParamValue(param){
        var paramValueText = $( "#"+param+"ValueText" )
        var newParamValueFromSlider= $( "#"+param+"Slider" )[0].value;
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/models/update_background_subtraction_param/'+param+'/'+newParamValueFromSlider,
            dataType: "json",
            success: function (data) {
                paramValueText.text( newParamValueFromSlider);
                console.log("/models/update_background_subtraction_param is done!") 
            },
            error: function (errMsg) {
                console.log(" ERROR /models/update_background_subtraction_param ") 
            }
        });  

    }
    
function initVideoStreamFrame(){
    var eventSource = new EventSource('/main_video_stream');
    eventSource.onmessage = function(event) {
        var result = JSON.parse(event.data);
        streamKeys=Object.keys(result)
        streamKeys.forEach(stream => {
            var videoFrame = $('#'+stream)
            videoFrame.attr("src", 'data:image/jpeg;base64,' + result[stream]);
        });        
    };
    // eventSource.onerror = (err) => {
    //     console.log("Stream ERROR :", err);
    //     console.error("Stream done :", err);
    //     eventSource.close();
    //   };

    videoInitialized=true;
}



function initData(){
    getObjectDetectionList();
    if (videoInitialized==false)
        initVideoStreamFrame()

    // src="{{ url_for('video_stream') }}"
}

function onClickReset(){
    toggleDisabledResetButton(true);
    videoInitialized=false;
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/stream/reset',
        dataType: "json",
        success: function (data) {
            console.log("/stream/reset")
            stopStreamOnReset()
            // if (videoInitialized==false)
            //     initVideoStreamFrame()
        },
        error: function (errMsg) {
            console.log(" ERROR IN reset")
            toggleDisabledResetButton(false)
        }
    });  
}


function stopStreamOnReset(){
    if(is_running_stream){
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(false);
        toggleDisabledLoadingModelButton(false);
        sendStopVideoRequest();
        is_running_stream=!is_running_stream
    }
    
} 

function onClickStartOfflineDetection(){

    toggleDisabledStartStopButton(true);
    toggleDisabledDetectionMethodSelect(true);
    toggleDisabledLoadingModelButton(true,showSpinner=false);
    toggleDisabledResetButton(true);

    $("#offlineDetectionButton").attr("disabled", true);
    $("#offlineDetectionButton").children().css( "display", "inline-block" )

    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_offline_detection',
        dataType: "json",
        success: function (data) {
            toggleDisabledStartStopButton(false);
            toggleDisabledDetectionMethodSelect(false);
            toggleDisabledLoadingModelButton(false,showSpinner=false);
            toggleDisabledResetButton(false);
            $("#offlineDetectionButton").attr("disabled", false);
            $("#offlineDetectionButton").children().css( "display", "none" )
            console.log("  start_offline_detection  success")
        },
        error: function (errMsg) {
            toggleDisabledStartStopButton(true);
            toggleDisabledDetectionMethodSelect(true);
            toggleDisabledLoadingModelButton(true,showSpinner=false);
            toggleDisabledResetButton(false);
            $("#offlineDetectionButton").attr("disabled", false);
            $("#offlineDetectionButton").children().css( "display", "none" )

            console.log(" ERROR IN start_offline_detection")
        }
    });  
}

function onClickToggleStopStart(){        
    if(is_running_stream){
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(false);
        toggleDisabledLoadingModelButton(false);
        toggleDisabledResetButton(false);
        sendStopVideoRequest();
    }else{
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(true);
        toggleDisabledLoadingModelButton(true,showSpinner=false);
        toggleDisabledResetButton(true);
        sendStartVideoRequest();
        // toggleDisabledResetButton(false)
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

function setModelNameText(text){
    var selectModelText = $("#selectModelText");
    selectModelText.text(text);
}

function setModelNameTextToLoadState(newSelectedModel){
    var selectModelText = $("#selectModelText");
    selectModelText.text(newSelectedModel + " est en cours de chargement ...");
}


function getObjectDetectionList(){
        $.ajax({
            type: "GET",
            url: $SCRIPT_ROOT + '/get_object_detection_list',
            dataType: "json",
            success: function (data) {
                fillobjectDetectionSelect(data);
            },
            error: function (errMsg) {
                console.log(" ERROR IN get_object_detection_list changing server ...")
                changeServerApi()
            },
            timeout:1200
        });   
    }


function onClickLoadModel(){
    toggleDisabledLoadingModelButton(true);
    toggleDisabledStartStopButton(true);
    toggleDisabledResetButton(true);
    
    if (selected_model_name==null){
        selected_model_name=$( "#objectDetectionSelect" )[0].value
    }    
    setModelNameTextToLoadState();
   
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/models/load/'+selected_model_name,
        dataType: "json",
        success: function (data) {
            console.log(" /models/load/"+selected_model_name)
            // clearInterval(intervalID);
            toggleDisabledLoadingModelButton(false);
            toggleDisabledStartStopButton(false);
            toggleDisabledResetButton(false);
            setModelNameText(""+selected_model_name +" est chargé correctement.") 
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
            setModelNameText("Error in loading "+selected_model_name +"!!")

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
    selectedVideo=$('#inputVideoFile').find(":selected").val();
      $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_stream/'+selectedVideo,
        dataType: "json",
        success: function (data) {
            console.log(" start_stream "+selectedVideo)
           
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

 




function onChangeObjectDetection(){
    console.log($( "#objectDetectionSelect" )[0].value );
    selected_model_name=$( "#objectDetectionSelect" )[0].value
    
}

function toggleDisabledResetButton(setToDisabled){
    if(setToDisabled){
        $("#resetButton").prop('disabled', true);
    }else{
        $("#resetButton").prop('disabled', false);
    }
}

function toggleDisabledDetectionMethodSelect(setToDisabled){
    if(setToDisabled){
        $("#objectDetectionSelect").prop('disabled', true);
    }else{
        $("#objectDetectionSelect").prop('disabled', false);
    }
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

function toggleDisabledBackgroundSubtractionStream(){
    if (showBackgroundSubtractionStream ){
        $("#accordionFlushBackgroundSubtraction").css("display", "block");
    }else{
        $("#accordionFlushBackgroundSubtraction").css("display", "none");
    }
}

function onClickSwitchTab(stream){
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/switch_client_stream/'+stream,
        dataType: "json",
        success: function (data) {
            console.log("  onClickSwitchTab  "+stream)
        },
        error: function (errMsg) {
            console.log(" error onClickSwitchTab true"+stream)
        }
    });  
}


function changeServerApi(){

    if  (secondApiIpChecked==false){
        if (api_server=='localhost'){
            api_server='raspberrypi.local'
        }
        else if (api_server=='raspberrypi.local'){
            api_server='localhost'
        }
        $SCRIPT_ROOT = "http://"+api_server+":8000/"
        secondApiIpChecked=true
        getObjectDetectionList()

    }

}
// function onCheckedBackgroundSubtractionService(){
//     showBackgroundSubtractionStream=$("#backgroundSubtractionCheckChecked").prop("checked")
//     toggleDisabledBackgroundSubtractionStream();

//     $.ajax({
//         type: "POST",
//         url: $SCRIPT_ROOT + '/background_subtraction/show_stream/'+showBackgroundSubtractionStream,
//         dataType: "json",
//         success: function (data) {
//             console.log("  toggleDisabledBackgroundSubtractionStream  ")
//         },
//         error: function (errMsg) {
//             console.log(" error toggleDisabledBackgroundSubtractionStream true")
//         }
//     });  

// }

// window.onbeforeunload = function (e) {
//     var e = e || window.event;
//     alert("clean_memory")
//     $.ajax({
//         type: "POST",
//         url: $SCRIPT_ROOT + '/clean_memory',
//         // The key needs to match your method's input parameter (case-sensitive).
//         // data: JSON.stringify({ Markers: markers }),
//         // contentType: "application/json; charset=utf-8",
//         dataType: "json",
//         success: function (data) {
//             console.log(" memory cleaned")
//             // alert("")
//             return e
//         },
//         error: function (errMsg) {
//             console.log(" ERROR IN memory cleaning")
//         }
//     });


// };
//  </script>