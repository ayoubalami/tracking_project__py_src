

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
var showMissingTracks=false;
var main_video_stream_error_count=0;

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


    function updateCNNDetectorParamValue(param){
        var newParamValueFromSlider= $( "#"+param+"_Slider" )[0].value;
        $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);

        if (param.startsWith('tracking_')){
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            param=param.substring(9)
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            $( "#"+param+"_Slider" )[0].value= newParamValueFromSlider;
        }else{
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"_ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"_Slider" )[0].value= newParamValueFromSlider;
        }
         
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/models/update_cnn_detector_param/'+param+'/'+newParamValueFromSlider,
            dataType: "json",
            success: function (data) {
                // paramValueText.text( newParamValueFromSlider);
                console.log("/models/update_cnn_detector_param is done!") 
            },
            error: function (errMsg) {
                console.log(" ERROR /models/update_cnn_detector_param ") 
            }
        });  
    }

    function updateBackgroundSubtractionParamValue(param){
        // tracking_varThreshold
        var newParamValueFromSlider= $( "#"+param+"Slider" )[0].value;

        if (param.startsWith('tracking_')){
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            param=param.substring(9)
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#"+param+"Slider" )[0].value= newParamValueFromSlider;
        }else{
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"Slider" )[0].value= newParamValueFromSlider;
        }

        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/models/update_background_subtraction_param/'+param+'/'+newParamValueFromSlider,
            dataType: "json",
            success: function (data) {
                // paramValueText.text( newParamValueFromSlider);
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
        main_video_stream_error_count=0
        var result = JSON.parse(event.data);
        streamKeys=Object.keys(result)
        streamKeys.forEach(stream => {
            var videoFrame = $('#'+stream)
            videoFrame.attr("src", 'data:image/jpeg;base64,' + result[stream]);
        });        
    };
    eventSource.onerror = (err) => {
        main_video_stream_error_count++;
        console.log("Stream ERROR :", err);
        console.error("Stream done :", err);
        if(main_video_stream_error_count>5)
            eventSource.close();
      };

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
    toggleDisabledNextFrameButton(true);
    $("#offlineDetectionButton").attr("disabled", true);
    $("#offlineDetectionButton").children().css( "display", "inline-block" )
    selectedVideo=$('#inputVideoFile').find(":selected").val();
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_offline_detection/'+selectedVideo,
        dataType: "json",
        success: function (data) {
            toggleDisabledStartStopButton(false);
            toggleDisabledDetectionMethodSelect(false);
            toggleDisabledLoadingModelButton(false,showSpinner=false);
            toggleDisabledResetButton(false);
            toggleDisabledNextFrameButton(false);
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
        toggleDisabledNextFrameButton(false)
        sendStopVideoRequest();
    }else{
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(true);
        toggleDisabledLoadingModelButton(true,showSpinner=false);
        toggleDisabledResetButton(true);
        toggleDisabledNextFrameButton(true)
        sendStartVideoRequest();

        // toggleDisabledResetButton(false)
    }
    is_running_stream=!is_running_stream
    
}

function fillobjectDetectionSelect(methodsList){
    var objectDetectionSelect = $("#objectDetectionSelect");
    var tracking_objectDetectionSelect = $("#tracking_objectDetectionSelect");

    console.log(methodsList);
    methodsList.forEach(method => {
        var el = document.createElement("option");
        el.textContent = method.name;
        el.value = method.name;
        objectDetectionSelect.append(el);
        el = document.createElement("option");
        el.textContent = method.name;
        el.value = method.name;
        tracking_objectDetectionSelect.append(el);
    });
}

function setModelNameText(text){
    $("#selectModelText").text(text);
    $("#tracking_selectModelText").text(text);
}

function setModelNameTextToLoadState(newSelectedModel){
    $("#selectModelText").text(newSelectedModel + " est en cours de chargement ...");
    $("#tracking_selectModelText").text(newSelectedModel + " est en cours de chargement ...");
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


function onClickLoadModel(source){
    toggleDisabledLoadingModelButton(true);
    toggleDisabledStartStopButton(true);
    toggleDisabledResetButton(true);
    toggleDisabledNextFrameButton(true);
    $("#offlineDetectionButton").attr("disabled", true);
    $("#offlineTrackingButton").attr("disabled", true);

    if (selected_model_name==null){
        if (source=='for_tracking')
            selected_model_name=$( "#tracking_objectDetectionSelect" )[0].value;
        else
            selected_model_name=$( "#objectDetectionSelect" )[0].value;
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
            toggleDisabledNextFrameButton(false);
            $("#offlineDetectionButton").attr("disabled", false);
            $("#offlineTrackingButton").attr("disabled", false);
        
            setModelNameText(""+selected_model_name +" est chargé correctement.") 
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
            setModelNameText("ERROR in loading "+selected_model_name +"!!")
            toggleDisabledLoadingModelButton(false);
            toggleDisabledStartStopButton(false);
            toggleDisabledResetButton(false);
            toggleDisabledNextFrameButton(false);
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

 
function onChangeObjectDetection(source){
    if (source=='for_tracking'){
        selected_model_name=$( "#tracking_objectDetectionSelect" )[0].value;
        $( "#objectDetectionSelect" )[0].value=selected_model_name
    }else{
        selected_model_name=$( "#objectDetectionSelect" )[0].value
        $( "#tracking_objectDetectionSelect" )[0].value=selected_model_name
    }
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
        $("#tracking_objectDetectionSelect").prop('disabled', true);
    }else{
        $("#objectDetectionSelect").prop('disabled', false);
        $("#tracking_objectDetectionSelect").prop('disabled', false);
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
        $("#tracking_loadModelButton").attr("disabled", true);
        
        if(showSpinner){
            $("#loadModelButton").children().css( "display", "inline-block" )
            $("#tracking_loadModelButton").children().css( "display", "inline-block" )
        }
        loadingDetectionModel=true;
    }else{
        $("#loadModelButton").attr("disabled", false);
        $("#loadModelButton").children().css( "display", "none" )
        $("#tracking_loadModelButton").attr("disabled", false);
        $("#tracking_loadModelButton").children().css( "display", "none" )
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
 
function sendGoToNextFrameRequest(){
    toggleDisabledNextFrameButton(true);
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/get_next_frame',
        dataType: "json",
        success: function (data) {
            console.log(" get_next_frame ")
            toggleDisabledNextFrameButton(false);
        },
        error: function (errMsg) {
            console.log(" ERROR IN start_stream")
            toggleDisabledNextFrameButton(false)
        }
    });    
}

function onClickGoToNextFrame(){        
    if(is_running_stream==false){
        sendGoToNextFrameRequest();
    }
}

function toggleDisabledNextFrameButton(setToDisabled){
    if (setToDisabled){
        $("#goToNextFrameButton").attr("disabled", true);
    }else{
        $("#goToNextFrameButton").attr("disabled", false);
    }
}

function onClickTrackWithDetection(){
    $("#tracking_accordionFlushDetectorParams").show();
    $("#tracking_accordionFlushBackgroundSubtraction").hide();
    send_track_with_request('cnn_detection')
}

function onClickTrackWithBackgroundSubtraction(){
    $("#tracking_accordionFlushDetectorParams").hide();
    $("#tracking_accordionFlushBackgroundSubtraction").show();
    send_track_with_request('background_subtraction')
}

function send_track_with_request(param){
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/track_with/'+param,
        dataType: "json",
        success: function (data) {
            console.log(" track_with ")
        },
        error: function (errMsg) {
            console.log(" ERROR IN start_stream")
        }
    });    
}

function onCheckedShowMissingTracks(){
    showMissingTracks=$("#showMissingTracksCheckbox")[0].checked;
    console.log(showMissingTracks);
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/show_missing_tracks/'+showMissingTracks,
        dataType: "json",
        success: function (data) {
            console.log(" show_missing_tracks ")
        },
        error: function (errMsg) {
            console.log(" ERROR IN show_missing_tracks")
        }
    });    
}

function onClickStartOfflineTracking(){

    toggleDisabledStartStopButton(true);
    toggleDisabledDetectionMethodSelect(true);
    toggleDisabledLoadingModelButton(true,showSpinner=false);
    toggleDisabledResetButton(true);
    toggleDisabledNextFrameButton(true)
    $("#offlineDetectionButton").attr("disabled", true);
    $("#offlineDetectionButton").children().css( "display", "inline-block" )
    $("#offlineTrackingButton").attr("disabled", true);
    $("#offlineTrackingButton").children().css( "display", "inline-block" )
    selectedVideo=$('#inputVideoFile').find(":selected").val();

    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_offline_tracking/'+selectedVideo,
        dataType: "json",
        success: function (data) {
            toggleDisabledStartStopButton(false);
            toggleDisabledDetectionMethodSelect(false);
            toggleDisabledLoadingModelButton(false,showSpinner=false);
            toggleDisabledResetButton(false);
            toggleDisabledNextFrameButton(false)
            $("#offlineDetectionButton").attr("disabled", false);
            $("#offlineDetectionButton").children().css( "display", "none" )
            $("#offlineTrackingButton").attr("disabled", false);
            $("#offlineTrackingButton").children().css( "display", "none" )
            console.log("  start_offline_tracking  success")
        },
        error: function (errMsg) {
            toggleDisabledStartStopButton(true);
            toggleDisabledDetectionMethodSelect(true);
            toggleDisabledLoadingModelButton(true,showSpinner=false);
            toggleDisabledResetButton(false);
            toggleDisabledNextFrameButton(false)
            $("#offlineDetectionButton").attr("disabled", false);
            $("#offlineDetectionButton").children().css( "display", "none" )
            $("#offlineTrackingButton").attr("disabled", false);
            $("#offlineTrackingButton").children().css( "display", "none" )
            console.log(" ERROR IN start_offline_tracking")
        }
    });  
}

function onCheckedActivateStreamSimulation(){
    activateStreamSimulation=$("#streamStimulationCheckbox")[0].checked;
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/activate_stream_simulation/'+activateStreamSimulation,
        dataType: "json",
        success: function (data) {
            console.log(" activate_stream_simulation ")
        },
        error: function (errMsg) {
            console.log(" ERROR IN activate_stream_simulation")
        }
    });
}

function onCheckedUseCNNFeatureExtraction(){
    useCNNFeatureExtractionCheckbox=$("#useCNNFeatureExtractionCheckbox")[0].checked;
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/use_cnn_feature_extraction_on_tracking/'+useCNNFeatureExtractionCheckbox,
        dataType: "json",
        success: function (data) {
            console.log(" activate_stream_simulation ")
        },
        error: function (errMsg) {
            console.log(" ERROR IN activate_stream_simulation")
        }
    });   
}

function onCheckedActivateDetection(){
    activateDetection=$("#activateDetectionCheckbox")[0].checked;
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/activate_detection/'+activateDetection,
        dataType: "json",
        success: function (data) {
            console.log(" activate_detection ")
        },
        error: function (errMsg) {
            console.log(" ERROR IN activate_detection")
        }
    });
}

function updateTrackingParamValue(param){
    var newParamValueFromSlider= $( "#"+param+"Slider" )[0].value;
    $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/update_tracking_param/'+param+'/'+newParamValueFromSlider,
        dataType: "json",
        success: function (data) {
            console.log("/update_tracking_param is done!")
        },
        error: function (errMsg) {
            console.log(" ERROR /update_tracking_param ")
        }
    });  
}