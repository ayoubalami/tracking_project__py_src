

// var $SCRIPT_ROOT = "http://Raspberrypi.local:8000/"
var $SCRIPT_ROOT = "http://"+api_server+":7070"
// var $SCRIPT_ROOT = "http://192.168.137.226:8000"

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


// function initData(){
$( document ).ready(function(){

    // $("#multiSelectClassesSelectAllCheckbox").click(function(){
    //     onSelectAllClassesForDetection();
    // });
    initMultiClassesSelect();
    
   
    initStreamSourceParams()
    getObjectDetectionList();
    if (videoInitialized==false)
        initVideoStreamFrame()
    
    setTimeout(function(){
        if (stream_source=='RASPBERRY_CAM'){
            onClickToggleStopStart();
        }
    }, 500); 
    initMouseClickEventForObjectTracking();


    // $(document).on('click','.select2-selection__clear',function(){
    //     select2-selection__clear
    //     alert(5);
    //     // onButtonRemoveAllClasses();
    // });


});


window.onbeforeunload = function(event){
    console.log("loading ....");
        $.ajax({
            type: "POST",
            url: $SCRIPT_ROOT + '/clean_memory',
            dataType: "json",
            success: function (data) {
                console.log(" memory cleaned")
                return true
            },
            error: function (errMsg) {
                console.log(" ERROR IN memory cleaning")
            }
        });    
    }

    function updateCNNDetectorParamValue(param){
        var newParamValueFromSlider= $( "#"+param+"_Slider" )[0].value;
        console.log(newParamValueFromSlider);
        $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
        
        if (param.startsWith('tracking_')){
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            param=param.substring(9)
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            $( "#"+param+"_Slider" )[0].value= newParamValueFromSlider;
        }else{
            $( "#"+param+"_ValueText" ).text( newParamValueFromSlider);
            if (param !='anchorCount'){
                $( "#tracking_"+param+"_ValueText" ).text( newParamValueFromSlider);
                $( "#tracking_"+param+"_Slider" )[0].value= newParamValueFromSlider;
            }
          
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
            $( "#hybridTracking_"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#hybridTracking_"+param+"Slider" )[0].value= newParamValueFromSlider;
        }
        else if (param.startsWith('hybridTracking_')){
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            param=param.substring(15)
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#"+param+"Slider" )[0].value= newParamValueFromSlider;
            $( "#tracking_"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"Slider" )[0].value= newParamValueFromSlider;
        }
        else{
            $( "#"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#tracking_"+param+"Slider" )[0].value= newParamValueFromSlider;
            $( "#hybridTracking_"+param+"ValueText" ).text( newParamValueFromSlider);
            $( "#hybridTracking_"+param+"Slider" )[0].value= newParamValueFromSlider;
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
    var eventSource = new EventSource('/main_video_stream?cache=' + Date.now());
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
   

function onClickReset(){
    toggleDisabledResetButton(true);
    videoInitialized=false;
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/reset_stream',
        dataType: "json",
        success: function (data) {
            console.log("/reset_stream")
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
    // selectedResolution=$('#inputVideoResolution').find(":selected").val();
     
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
        toggleDisabledVideoFileButton(false)
        sendStopVideoRequest();
    }else{
        toggleDisabledStartStopButton(true);
        toggleDisabledDetectionMethodSelect(true);
        toggleDisabledLoadingModelButton(true,showSpinner=false);
        toggleDisabledResetButton(true);
        toggleDisabledNextFrameButton(true)
        toggleDisabledVideoFileButton(true)
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
            $("#inputVideoSecond_slider").attr("disabled", false);        

            toggleDisabledStartStopButton(false);
            // return e
        },
        error: function (errMsg) {
            console.log(" ERROR IN stop_stream")
            toggleDisabledStartStopButton(false);
        }
    });
}

function onChangeVideoResolutionSlider(detector_service){

    sliderName=''
    inputValueName=''
    if (detector_service=='BS'){
        sliderName='#BSinputVideoResolution_slider'
        inputValueName='#BSinputVideoResolution_ValueText'
    }
    if  (detector_service=='CNN'){
        sliderName='#CNNinputVideoResolution_slider'
        inputValueName='#CNNinputVideoResolution_ValueText'
    }

    var selectedResolution= $( sliderName )[0].value;
    $( inputValueName).text( selectedResolution+" %");
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/'+detector_service+'_set_video_resolution/'+selectedResolution,
        dataType: "json",
        success: function (data) {
            console.log(" video_resolution_from_second "+selectedResolution)
        },
        error: function (errMsg) {
            console.log(" ERROR IN video_resolution_from_second")
        }
    });  
}

function onChangeVideoSecondSlider(){
    var selectedSecond= $( "#inputVideoSecond_slider" )[0].value;
    // $( "#inputVideoSecond_ValueText" ).text( selectedSecond+"%");
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/set_video_starting_second/'+selectedSecond,
        dataType: "json",
        success: function (data) {
            console.log(" video_start_from_second "+selectedVideo)
        },
        error: function (errMsg) {
            console.log(" ERROR IN inputVideoSecond_ValueText")
        }
    });    
} 

function sendStartVideoRequest(){
    selectedVideo=$('#inputVideoFile').find(":selected").val();
    // var selectedResolution= $( "#inputVideoResolution_slider" )[0].value;
    // $( "#"+param+"ValueText" ).text( selectedResolution+" %");

    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/start_stream/'+selectedVideo,
        dataType: "json",
        success: function (data) {
            console.log(" start_stream "+selectedVideo)
            $('#startStopButton').html( 'Stop');
            $('#startStopButton').removeClass("btn-success");
            $('#startStopButton').addClass("btn-danger");
            $("#inputVideoSecond_slider").attr("disabled", true);        

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

function toggleDisabledVideoFileButton(setToDisabled){
    if(setToDisabled){
        $("#inputVideoFile").prop('disabled', true);
    }else{
        $("#inputVideoFile").prop('disabled', false);
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
        // $("#inputVideoSecond_slider").attr("disabled", true);        
        loadingStartStopButton=true;
    }else{
        $("#startStopButton").attr("disabled", false);
        // $("#inputVideoSecond_slider").attr("disabled", false);

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

function initStreamSourceParams(){
    console.log(stream_source);
    if (stream_source=='RASPBERRY_CAM'){
        $( "#videoFilesSelect" ).attr('style','display:none !important');
        $( "#RaspberryServoCameraInput" ).attr('style','display:flex !important');
    }
    else{
        $( "#videoFilesSelect" ).attr('style','display:flex !important');
        $( "#RaspberryServoCameraInput" ).attr('style','display:none !important');
    }
}


function updateServoMotorValue(axis){
    var newDegreeFromSlider= $( "#"+axis+"_axisServoDegree_slider" )[0].value;
    $( "#"+axis+"_axisServoDegree_valueText" ).text( newDegreeFromSlider);
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/rotate_servo_motor/'+axis+'/'+newDegreeFromSlider,
        dataType: "json",
        success: function (data) {
            console.log("/rotate_servo_motor is done!")
        },
        error: function (errMsg) {
            console.log(" ERROR /rotate_servo_motor ")
        }
    });  
}


function updateRaspberryCameraZoomValue(){
    var newZoomFromSlider= $( "#cameraZoom_slider" )[0].value;
    $( "#cameraZoom_valueText" ).text( newZoomFromSlider);
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/update_raspberry_camera_zoom/'+newZoomFromSlider,
        dataType: "json",
        success: function (data) {
            console.log("/update_raspberry_camera_zoom is done!")
        },
        error: function (errMsg) {
            console.log(" ERROR /update_raspberry_camera_zoom ")
        }
    });  
}

function initMouseClickEventForObjectTracking(){
    $('#trackingStream_1').click(function(event) {
        // Get the mouse click position
        var x = event.pageX - $(this).offset().left;
        var y = event.pageY - $(this).offset().top;
        var width = $(this).width();
        var height = $(this).height();
        console.log("origin width : " + x + ", height: " + y);
        console.log("ratio width : " + (x/width).toFixed(4) + ",ratio height: " + (y/height).toFixed(4));
        x=(x/width).toFixed(4);
        y=(y/height).toFixed(4);
        updateTrackedCoordinates(x,y)
    });
}

function updateTrackedCoordinates(x,y){
     $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/update_tracked_coordinates/'+x+'/'+y,
        dataType: "json",
        success: function (data) {
            console.log("/update_tracked_coordinates is done!")
        },
        error: function (errMsg) {
            console.log(" ERROR /update_tracked_coordinates ")
        }
    });  
}

function initMultiClassesSelect(){
    var multiSelectClasses= $( '#multiSelectClasses' );
    var tracking_multiSelectClasses= $('#tracking_multiSelectClasses');

    tracking_multiSelectClasses.select2({
        theme: "bootstrap-5",
        width: $( this ).data( 'width' ) ? $( this ).data( 'width' ) : $( this ).hasClass( 'w-100' ) ? '100%' : 'style',
        placeholder: $( this ).data( 'placeholder' ),
        closeOnSelect: false,
    });

    multiSelectClasses.select2({
        theme: "bootstrap-5",
        width: $( this ).data( 'width' ) ? $( this ).data( 'width' ) : $( this ).hasClass( 'w-100' ) ? '100%' : 'style',
        placeholder: $( this ).data( 'placeholder' ),
        closeOnSelect: false,
    });
 
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/get_class_labels',
        dataType: "json",
        success: function (classLabels) {
            classLabels.forEach(_class => {
                var el = document.createElement("option");
                el.textContent = _class.label;
                el.value = _class.id;
                multiSelectClasses.append(el);
                var el = document.createElement("option");
                el.textContent = _class.label;
                el.value = _class.id;
                tracking_multiSelectClasses.append(el);
            });
            console.log("/get_class_labels is done!")

        },
        error: function (errMsg) {
            console.log(" ERROR /get_class_labels ")
        }
    });  
}

function onClickMultiSelectClasses(id){
    var multiSelectClasses= $( '#'+id )
    var selectedIdx=multiSelectClasses.val()
    if (selectedIdx.length==0)
        selectedIdx=-1
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/set_selected_classes/'+selectedIdx,
        dataType: "json",
        success: function () {
            console.log(" /set_selected_classes is done")
        },
        error: function () {
            console.log(" ERROR /set_selected_classes ")
        }
    });  
}


function onVideoFileSelected(event){
    // selectedVideo=$('#inputVideoFile').find(":selected").val();
    // alert(event.target.value)
    $.ajax({
        type: "POST",
        url: $SCRIPT_ROOT + '/change_video_file/'+event.target.value,
        dataType: "json",
        success: function () {
            console.log(" /set_selected_classes is done")
        },
        error: function () {
            console.log(" ERROR /set_selected_classes ")
        }
    });  
}
 
