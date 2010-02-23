
/*
On Browser with no native JSON parsing the large hinet viualisation structure
can cause problems (e.g. in Firefox 3.0 it will fail to load). You could work
around this by using eval, accepting possible security problems.

On the checkbox issue: this is caused by XHTML attributes are not synched
to the actual properties, so you have to set those
http://dev.jquery.com/ticket/4283
http://elegantcode.com/2009/02/02/aspnet-checkbox-and-jquery/

using the .html function might also cause problems, but .append should do
the job 
*/


// stores the layer types provided by the server
// the properties are lists with the parameter names
var layer_type_params;

// variables storing the information about the current network
var layer_divs = [];  // store the div ids
var layer_types = [];

var cov_throbber_ids = [];
var cov_active_throbber_id;
var active_coverage_ids = [];  // contains id's of active elements

var benchmark = false;  // display benchmark information


// receive the parameter information for the layer types
function receive_layer_params(json_object) {
    var layer_names = json_object.result.layer_names;  // used for order
    layer_type_params = json_object.result.layer_params;
    $("#layer_type").empty();
    var i;
    for (i = 0; i < layer_names.length; i += 1) {
        $("#layer_type").append(
            $("<option></option>").val(layer_names[i]).html(layer_names[i])
        );
    }
    $("#add_layer").removeAttr('disabled');
}

// receive the names of the saved configs available on the server
function reveive_available_hinet_configs(json_object) {
    $("#hinet_available_config").empty();
    var config_names = json_object.result;
    var i;
    for (i = 0; i < config_names.length; i += 1) {
        $("#hinet_available_config").append(
            $("<option></option>").val(config_names[i]).html(config_names[i])
        );
    }
}

function request_hinet_config() {
    $("#layer_controls").html("<img src='/images/throbber.gif' />");
    $("#hinet_view").empty();
    $("#hinet_coverage").empty();
    $("#hinet_config").empty();
    var config_name = $('#hinet_available_config').val();
    $("#config_name").val(config_name);
    var json_object = {"method": "get_hinet_config",
                       "params": [config_name],
                       "id": "frontend"};
    var json_string = JSON.stringify(json_object);
    $.post("frontend.xhtml", json_string, receive_hinet_config,
           "json");
}

// receive one hinet config from the server and update the UI
function receive_hinet_config(json_object) {
    // reset all current layer
    layer_divs = [];
    layer_types = [];
    $("#layer_controls").empty();
    // set image size
    if ($("#square_image").is(':checked')) {
        $("#square_image").get(0).checked = false;
    }
    var image_size = json_object.result.image_size;
    $("#image_size_x").val(image_size[0]);
    $("#image_size_y").val(image_size[1]);
    if (image_size[0] === image_size[1]) {
        $("#square_image").get(0).checked = true;
    }
    $("#square_image").change();
    // build up new network
    var config = json_object.result.layer_configs;
    var layer_type;
    var layer_params;
    var param_name;
    var param_id;
    var i_param;
    var all_square;
    var i;
    for (i = 0; i < config.length; i += 1) {
        layer_type = config[i].layer_type;
        delete config[i].layer_type;
        $('#layer_type').val(layer_type);
        add_layer();
        // set text fields
        all_square = true;
        layer_params = layer_type_params[layer_type];
        for (i_param = 0; i_param < layer_params.length; i_param += 1) {
            param_name = layer_params[i_param];
            if (ends_with(param_name, "_xy")) {
                param_id = param_name.substr(0, param_name.length - 3) + "_" +
                           (i+1).toString();
                $("#" + param_id + "_x").val(config[i][param_name][0]);
                $("#" + param_id + "_y").val(config[i][param_name][1]);
                if (config[i][param_name][0] !== config[i][param_name][1]) {
                    all_square = false;
                }
            } else {
                param_id = param_name + "_" + (i+1).toString();
                $("#" + param_id).val(config[i][param_name]);
            }
        }
        if (!all_square) {
            $("#square_switchboard_"+(i+1).toString()).get(0).checked = false; 
        $("#square_switchboard_"+(i+1).toString()).change();
        }
    }
}

// send current network config to the server to save it there
function save_hinet_config() {
    var json_object = {"method": "save_hinet_config",
                       "params": [get_hinet_config(), $("#config_name").val()],
                       "id": "frontend"};
    var json_string = JSON.stringify(json_object);
    $.post("frontend.xhtml", json_string, reveive_available_hinet_configs,
           "json");
}

// request a new network based on the current parameters
function request_hinet() {
    $("#hinet_view").html("<img src='/images/throbber.gif' />");
    $("#hinet_coverage").empty();
    $("#hinet_config").empty();
    var json_object = {"method": "get_hinet",
                       "params": [get_hinet_config()],
                       "id": "frontend"};
    var json_string = JSON.stringify(json_object);
	// Request text and do the JSON parsing manually,
	// the problem is that jQuery does some sanitizing first, which leads
	// to problems with large structures in Firefox
	// ('script stack space quota is exhausted', line 507).
    $.post("frontend.xhtml", json_string, receive_hinet, "text");
}

// update the UI with the received network
function receive_hinet(json_text) {
	json_object = JSON.parse(json_text);
    if (json_object.hasOwnProperty('error')) {
        $("#hinet_view").html('<span class="error">' +
                              json_object.error + '</span>');
        return;
    }
    // update hinet view
    var hinet_view_html = json_object.result.html_view;
    $("#hinet_view").html(hinet_view_html).hide().fadeIn();
    // update hinet coverage
    $("#hinet_coverage").append('<h4>channel coverage</h4>');
    $("#hinet_coverage").append(
        '<div id="coverage_floats" class="float_container"></div>');
    $("#hinet_coverage").append(
        '<span class="note">Click on field square to display coverage.</span>');
    $("#hinet_coverage").append(
        '<br /><span class="timer"></span>');
    var hinet_coverage_svgs = json_object.result.hinet_coverage_svgs;
    var hinet_coverage_ids = json_object.result.hinet_coverage_ids;
    $("#coverage_floats").append([
            '<div class="layer_coverage">',
            'image <br />',
            hinet_coverage_svgs[0],
            '</div>'
        ].join("\n"));
    cov_throbber_ids = [];
    var i;
    var j;
    var cov_throbber_id;
    for (i = 1; i < hinet_coverage_svgs.length; i += 1) {
        cov_throbber_id = "cov_throbber_" + i.toString();
        cov_throbber_ids.push(cov_throbber_id);
        $("#coverage_floats").append([
            '<div class="layer_coverage">',
            'layer ' + i.toString() +
            '&#160;<img id="' + cov_throbber_id +
            '" src="/images/small_throbber.gif" class="hidden" /><br />',
            hinet_coverage_svgs[i],
            '</div>'
        ].join("\n"));
        // make the coverage svg elements clickable
        for (j = 0; j < hinet_coverage_ids[i].length; j += 1) {
            $("#" + hinet_coverage_ids[i][j]).click(
                _get_coverage_function(i,j));
        }
    }
    // update hinet params
    $("#hinet_config").append('<h4>network parameters</h4>');
    $("#hinet_config").append('<pre>' + json_object.result.hinet_config_str +
                              '</pre>');
}

function request_coverage(layer, channel) {
    cov_active_throbber_id = cov_throbber_ids[layer - 1];
    $("#" + cov_active_throbber_id).removeAttr("class");
    // send request
    var json_object = {"method": "get_layer_coverage",
                       "params": [layer, channel],
                       "id": "frontend"};
    var json_string = JSON.stringify(json_object);
    $.post("frontend.html", json_string, receive_coverage, "json");
}
    
function receive_coverage(json_object){
    var i;
    var start_time = (new Date).getTime();
    // reset color of the old coverage ids
    for (i in active_coverage_ids) {
        $("#" + active_coverage_ids[i]).attr("class", "cov_normal");
    }
    // change color of the covered ids
    active_coverage_ids = json_object.result;
    for (i in active_coverage_ids) {
        $("#" + active_coverage_ids[i]).attr("class", "cov_active");
    }
    $("#" + cov_active_throbber_id).attr("class", "hidden");
    if (benchmark) {
        var time_diff = new Date;
        time_diff.setTime((new Date).getTime() - start_time);
        $("span.timer").html("render time: " +
             time_diff.getMilliseconds().toString() +
    	     " ms");
    }
}

// add a layer in the GUI with the parameter fields
function add_layer() {
    var layer_num = (layer_divs.length + 1).toString();
    var layer_name = "layer_" + layer_num;
    var layer_type = $('#layer_type').val();
    layer_divs.push(layer_name);
    var layer_html = [
        '<div id="' + layer_name + '" class="single_layer_controls">',
        '<span style="font-style: italic;">layer ' + 
            layer_num + ' (' + layer_type + ')' +
            '</span>&#160;&#160;'
    ];
    var checkbox_name = "square_switchboard_" + layer_num;
    layer_html.push('<input type="checkbox" id="' + checkbox_name +
                    '" checked="checked" />');
    layer_html.push('<label for="'  + checkbox_name + '" id="' + checkbox_name
                    + '_label">square</label>&#160;&#160;');
    layer_types.push(layer_type);
    var layer_params = layer_type_params[layer_type];
    var i_param;
    var param_name;
    var param_id;
    var sync_fields = [];
    for (i_param = 0; i_param < layer_params.length; i_param += 1) {
        param_name = layer_params[i_param];
        if (ends_with(param_name, "_xy")) {
            param_name = param_name.substr(0, param_name.length - 3);
            param_id = param_name + "_" + layer_num;
            layer_html = layer_html.concat([
                '<label for="' + param_id + '_x">' + param_name + ': </label>',
                '<input type="text" id="' + param_id + '_x" value="1"' +
                ' class="number" />',
                '<input type="text" id="' + param_id + '_y" value="1"' +
                ' class="number" />',
                '&#160;&#160;'
            ]);
            sync_fields.push(param_id);
        } else {
            param_id = param_name + "_" + layer_num;
            layer_html = layer_html.concat([
                '<label for="' + param_id + '">' + param_name + ': </label>',
                '<input type="text" id="' + param_id + '" value="1"' +
                ' class="number" />',
                '&#160;&#160;'
            ]);
        }
    }
    layer_html = layer_html.concat(['</div>']);
    $("#layer_controls").append(layer_html.join("\n"));
    $("#" + layer_name).hide().slideDown('normal');
    if (sync_fields.length > 0) {
        $("#" + checkbox_name).change(function(){
            var i;
            for (i in sync_fields) {
                if ($(this).is(':checked')) {
                    $("#" + sync_fields[i] + "_x").keyup(
                        _get_sync_function(sync_fields[i]));
                    $("#" + sync_fields[i] + "_y").attr('disabled', true);
                }
                else {
                    $("#" + sync_fields[i] + "_x").unbind();
                    $("#" + sync_fields[i] + "_y").removeAttr('disabled');
                }
            }
        });
        $("#" + checkbox_name).change();
    } else {
        $("#" + checkbox_name).remove();
        $("#" + checkbox_name + "_label").remove();
    }
}

// remove the last layer
function remove_layer() {
    $("#" + layer_divs.pop()).slideUp('normal',
                                      function() { $(this).remove(); });
    layer_types.pop();
}

// extract the hinet config from the form fields
function get_hinet_config() {
    var image_size_x = parseInt($("#image_size_x").val());
    var image_size_y = parseInt($("#image_size_y").val());
    var layer_configs = [];
    var layer_data = {};
    var layer_num;
    var layer_type;
    var layer_params;
    var i_layer;
    var i_param;
    for (i_layer = 0; i_layer < layer_divs.length; i_layer += 1) {
        layer_num = (i_layer + 1).toString();
        layer_type = layer_types[i_layer]
        layer_params = layer_type_params[layer_type];
        layer_data = {"layer_type": layer_type};
        for (i_param = 0; i_param < layer_params.length; i_param += 1) {
            param_name = layer_params[i_param];
            if (ends_with(param_name, "_xy")) {
                param_id = param_name.substr(0, param_name.length - 3) + "_" +
                           (i_layer+1).toString();
                layer_data[param_name] = [
                    parseInt($("#" + param_id + "_x").val()),
                    parseInt($("#" + param_id + "_y").val())
                ];
            } else {
                param_id = param_name + "_" + layer_num;
                layer_data[param_name] = parseInt($("#" + param_id).val());
            }
        }
        layer_configs.push(layer_data);
    }
    return {
        "image_size": [image_size_x, image_size_y],
        "layer_configs": layer_configs
    }
}

// helper function to capture variables in closure for callback function

function _get_sync_function(field_name) {
    return function(){ 
        $("#" + field_name + "_y").val($("#" + field_name + "_x").val()); 
    }
}

function _get_coverage_function(i,j) {
    return function(){ 
        request_coverage(i,j);
    }
}

// helper function to check if the string ends with the pattern string
function ends_with(thestring, pattern) {
    var d = thestring.length - pattern.length;
    return d >= 0 && thestring.lastIndexOf(pattern) === d;
}


$(function() {
    // register controls
    $("#load_config").click(request_hinet_config);
    $("#save_config").click(save_hinet_config);
    $("#show_hinet").click(request_hinet);
    $("#add_layer").click(add_layer).attr('disabled', true);
    $("#remove_layer").click(remove_layer);
    $("#square_image").change(function(){
        if ($(this).is(':checked')) {
            $("#image_size_x").keyup(_get_sync_function("image_size"));
            $("#image_size_y").attr('disabled', true);
        } else {
            $("#image_size_x").unbind();
            $("#image_size_y").removeAttr('disabled');
        }
    })
    $("#square_image").change();
    // get the different layer types and their parameters
    var json_object = {"method": "get_layer_params",
                       "id": "frontend"};
    var json_string = JSON.stringify(json_object);
    $.post("frontend.xhtml", json_string, receive_layer_params, "json");
    json_object = {"method": "get_available_hinet_configs",
                   "id": "frontend"};
    json_string = JSON.stringify(json_object);
    $.post("frontend.xhtml", json_string, reveive_available_hinet_configs,
           "json");
    // preload images
    var img = new Image();
    img.src = '/images/throbber.gif';
    img.src = '/images/small_throbber.gif';
});